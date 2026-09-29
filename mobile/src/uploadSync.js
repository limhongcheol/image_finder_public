// ─────────────────────────────────────────────────────────
//  uploadSync.js — 갤러리 업로드 핵심 로직 (UI + 백그라운드 공용)
//
//  syncNewMedia(): 갤러리에서 새 사진/동영상만 골라 640 리사이즈(이미지) + 메타데이터로 업로드.
//  백그라운드 태스크: 앱이 꺼져 있어도 OS 가 허락할 때 주기적으로 새 사진 업로드.
//   ⚠️ 모바일 OS 한계: '완전 종료 + 즉시'는 불가. ~15분 간격 best-effort (특히 iOS).
// ─────────────────────────────────────────────────────────
import { Platform } from 'react-native';
import * as MediaLibrary from 'expo-media-library/legacy';
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as LegacyFS from 'expo-file-system/legacy';
import { zip } from 'react-native-zip-archive';
import BackgroundService from 'react-native-background-actions';
import { uploadMedia, uploadBatchZip } from './api';
import { log } from './logger';

export const UID_KEY = 'lumina_user_id';
export const upKey = (uid) => `lumina_uploaded_${uid}`;

// 갤러리에서 안 올린 미디어를 골라 업로드. UI/백그라운드 양쪽에서 호출.
//   opts: { limit, onProgress({done,total,ok,fail,current}), isCancelled() }
export async function syncNewMedia(userId, opts = {}) {
  const { limit = 0, onProgress, isCancelled } = opts;
  const uid = (userId || '').trim();
  if (!uid) throw new Error('user_id 없음');

  const perm = await MediaLibrary.requestPermissionsAsync();
  if (!perm.granted) throw new Error('갤러리 권한 없음');

  // 이미 올린 asset id 집합 (중복 방지)
  let uploaded = new Set();
  try {
    const raw = await AsyncStorage.getItem(upKey(uid));
    if (raw) uploaded = new Set(JSON.parse(raw));
  } catch {}

  // 사진만 수집 (동영상 제외 — Cloudflare 100MB 업로드 제한)
  let assets = [];
  let after;
  let hasNext = true;
  while (hasNext) {
    const page = await MediaLibrary.getAssetsAsync({
      mediaType: MediaLibrary.MediaType.photo,
      first: 1000, after,
    });
    assets = assets.concat(page.assets);
    after = page.endCursor; hasNext = page.hasNextPage;
  }

  let todo = assets.filter(a => !uploaded.has(a.id));
  if (limit > 0) todo = todo.slice(0, limit);

  let ok = 0, fail = 0, lastErr = '';
  const BATCH_SIZE = 100; // 한 번에 올릴 병렬 처리 개수

  for (let i = 0; i < todo.length; i += BATCH_SIZE) {
    if (isCancelled && isCancelled()) break;
    const chunk = todo.slice(i, i + BATCH_SIZE);

    // 현재 진행 상태 표시 (청크의 첫 번째 파일 이름 기준)
    onProgress && onProgress({
      done: i,
      total: todo.length,
      ok, fail,
      current: chunk[0]?.filename || ''
    });

    // 100장을 병렬로 처리
    const results = await Promise.all(chunk.map(async (a) => {
      try {
        let info = {};
        try { info = await MediaLibrary.getAssetInfoAsync(a); } catch {}
        const meta = {
          captured_at: a.creationTime ? new Date(a.creationTime).toISOString() : '',
          media_type: 'image',
          lat: info.location ? info.location.latitude : undefined,
          lon: info.location ? info.location.longitude : undefined,
          asset_id: a.id,                       // 오프라인 번들용 식별자
          asset_uri: info.localUri || a.uri,
          platform: Platform.OS,
        };
        const srcUri = info.localUri || a.uri;

        // 640 리사이즈 → 항상 100MB 이하 (이미지만 업로드)
        const ctx = ImageManipulator.manipulate(srcUri);
        ctx.resize({ width: 640, height: 640 });
        const rendered = await ctx.renderAsync();
        const out = await rendered.saveAsync({ compress: 0.85, format: SaveFormat.JPEG });

        await uploadMedia(out.uri, a.filename, uid, meta);
        return { success: true, id: a.id };
      } catch (e) {
        return { success: false, error: e };
      }
    }));

    // 병렬 처리 결과 집계
    results.forEach(r => {
      if (r.success) {
        uploaded.add(r.id);
        ok++;
      } else {
        fail++;
        if (!lastErr) lastErr = String(r.error?.message || r.error);
      }
    });

    try { await AsyncStorage.setItem(upKey(uid), JSON.stringify([...uploaded])); } catch {}
    onProgress && onProgress({ done: Math.min(i + BATCH_SIZE, todo.length), total: todo.length, ok, fail });
  }
  try { await AsyncStorage.setItem(upKey(uid), JSON.stringify([...uploaded])); } catch {}

  return { ok, fail, lastErr, cancelled: !!(isCancelled && isCancelled()) };
}

// ─────────────────────────────────────────────────────────
//  배치 업로드: 640 리사이즈본 여러 장을 zip 1개(≤95MB)로 묶어 /upload_batch 로 전송.
//  1장씩 1000번 요청 → zip 1번 요청으로 줄여 대폭 빠름. (네이티브 zip = 디스크 기반, 메모리 안전)
// ─────────────────────────────────────────────────────────
const BATCH_DIR = LegacyFS.cacheDirectory + 'lumina_batch/';
const MAX_BATCH_BYTES = 95 * 1024 * 1024;   // 95MB (Cloudflare 100MB 안전 마진)
const noScheme = (uri) => (uri || '').replace(/^file:\/\//, '');

export async function syncNewMediaBatch(userId, opts = {}) {
  const { limit = 0, onProgress, isCancelled } = opts;
  const uid = (userId || '').trim();
  if (!uid) throw new Error('user_id 없음');

  const perm = await MediaLibrary.requestPermissionsAsync();
  if (!perm.granted) throw new Error('갤러리 권한 없음');

  let uploaded = new Set();
  try {
    const raw = await AsyncStorage.getItem(upKey(uid));
    if (raw) uploaded = new Set(JSON.parse(raw));
  } catch {}

  // 사진만 수집 (동영상 제외)
  let assets = [];
  let after;
  let hasNext = true;
  while (hasNext) {
    const page = await MediaLibrary.getAssetsAsync({
      mediaType: MediaLibrary.MediaType.photo, first: 1000, after,
    });
    assets = assets.concat(page.assets);
    after = page.endCursor; hasNext = page.hasNextPage;
  }
  let todo = assets.filter(a => !uploaded.has(a.id));
  if (limit > 0) todo = todo.slice(0, limit);
  const total = todo.length;

  let ok = 0, fail = 0, lastErr = '';
  let batchFiles = [];   // 이번 zip 에 들어갈 파일들(jpg + .meta.json)
  let batchIds = [];
  let batchBytes = 0;
  let prepared = 0;

  // 작업 폴더 초기화
  try { await LegacyFS.deleteAsync(BATCH_DIR, { idempotent: true }); } catch {}
  await LegacyFS.makeDirectoryAsync(BATCH_DIR, { intermediates: true });

  const flush = async () => {
    if (!batchFiles.length) return;
    const n = batchIds.length;
    const zipPlain = noScheme(LegacyFS.cacheDirectory) + `batch_${Date.now()}.zip`;
    onProgress && onProgress({ done: prepared, total, ok, fail, current: `전송중 (${n}장)…`, phase: 'send', batchBytes, maxBytes: MAX_BATCH_BYTES });
    try {
      await zip(batchFiles.map(noScheme), zipPlain, 0);   // 0 = 저장(무압축, 빠름; jpg는 어차피 압축됨)
      await uploadBatchZip('file://' + zipPlain, uid, n);
      batchIds.forEach(id => uploaded.add(id));
      ok += n;
      await AsyncStorage.setItem(upKey(uid), JSON.stringify([...uploaded]));
      onProgress && onProgress({ done: prepared, total, ok, fail, current: `전송 완료 (${n}장)`, phase: 'sent', batchBytes: MAX_BATCH_BYTES, maxBytes: MAX_BATCH_BYTES });
    } catch (e) {
      fail += n;
      if (!lastErr) lastErr = String(e?.message || e);
    } finally {
      try { await LegacyFS.deleteAsync('file://' + zipPlain, { idempotent: true }); } catch {}
      for (const f of batchFiles) { try { await LegacyFS.deleteAsync(f, { idempotent: true }); } catch {} }
      batchFiles = []; batchIds = []; batchBytes = 0;
    }
    // 배치 비움 → 진행도바 초기화(0)
    onProgress && onProgress({ done: prepared, total, ok, fail, phase: 'compress', batchBytes: 0, maxBytes: MAX_BATCH_BYTES });
  };

  for (let i = 0; i < todo.length; i++) {
    if (isCancelled && isCancelled()) break;
    const a = todo[i];
    onProgress && onProgress({ done: prepared, total, ok, fail, current: a.filename || '', phase: 'compress', batchBytes, maxBytes: MAX_BATCH_BYTES });
    try {
      let info = {};
      try { info = await MediaLibrary.getAssetInfoAsync(a); } catch {}
      const srcUri = info.localUri || a.uri;

      const ctx = ImageManipulator.manipulate(srcUri);
      ctx.resize({ width: 640, height: 640 });
      const rendered = await ctx.renderAsync();
      const out = await rendered.saveAsync({ compress: 0.85, format: SaveFormat.JPEG });

      const name = (a.filename || `photo_${a.id}.jpg`).replace(/[\\/]/g, '_');
      const imgPath = BATCH_DIR + name;
      await LegacyFS.copyAsync({ from: out.uri, to: imgPath });
      try { await LegacyFS.deleteAsync(out.uri, { idempotent: true }); } catch {}

      const meta = {
        original_filename: name,
        captured_at: a.creationTime ? new Date(a.creationTime).toISOString() : '',
        media_type: 'image',
        lat: info.location ? info.location.latitude : null,
        lon: info.location ? info.location.longitude : null,
        asset_id: a.id,                       // 오프라인 번들용 식별자
        asset_uri: info.localUri || a.uri,
        platform: Platform.OS,
      };
      const metaPath = imgPath + '.meta.json';
      await LegacyFS.writeAsStringAsync(metaPath, JSON.stringify(meta));

      const sz = (await LegacyFS.getInfoAsync(imgPath, { size: true })).size || 0;
      // 한도 초과면 현재 사진 추가 전에 먼저 보냄
      if (batchBytes + sz > MAX_BATCH_BYTES && batchFiles.length) {
        await flush();
      }
      batchFiles.push(imgPath, metaPath);
      batchIds.push(a.id);
      batchBytes += sz;
      prepared++;
    } catch (e) {
      fail++;
      if (!lastErr) lastErr = String(e?.message || e);
    }
  }
  if (!(isCancelled && isCancelled())) await flush();

  try { await LegacyFS.deleteAsync(BATCH_DIR, { idempotent: true }); } catch {}
  return { ok, fail, lastErr, cancelled: !!(isCancelled && isCancelled()) };
}

// ─────────────────────────────────────────────────────────
//  포그라운드 서비스 업로드 — 상시 알림 띄우고 앱을 닫아도(백그라운드) 계속 전송.
//  syncFn = syncNewMedia | syncNewMediaBatch
// ─────────────────────────────────────────────────────────
const FG_OPTIONS = {
  taskName: 'LuminaUpload',
  taskTitle: '📦 압축중…',
  taskDesc: '업로드 준비 중',
  taskIcon: { name: 'ic_launcher', type: 'mipmap' },
  color: '#c85a30',
  foregroundServiceType: ['dataSync'],
  progressBar: { max: 100, value: 0 },
};

export async function runForegroundUpload(userId, syncFn, opts = {}) {
  const { limit = 0, onProgress, onDone } = opts;
  const task = async () => {
    let result = { ok: 0, fail: 0, lastErr: '' };
    try {
      result = await syncFn(userId, {
        limit,
        onProgress: async (p) => {
          onProgress && onProgress(p);
          try {
            const max = p.maxBytes || (95 * 1024 * 1024);
            const pct = Math.max(0, Math.min(100, Math.round((p.batchBytes || 0) / max * 100)));
            const title =
              p.phase === 'send' ? '📤 전송중…' :
              p.phase === 'sent' ? '✅ 전송 완료' :
              '📦 압축중…';
            await BackgroundService.updateNotification({
              taskTitle: title,
              taskDesc: `전송 성공 ${p.ok}/${p.total}장${p.fail ? ` · 실패 ${p.fail}` : ''}`,
              progressBar: { max: 100, value: pct, indeterminate: p.phase === 'send' },
            });
          } catch {}
        },
        isCancelled: () => !BackgroundService.isRunning(),
      });
    } catch (e) {
      result = { ok: 0, fail: 0, lastErr: String(e?.message || e) };
    } finally {
      onDone && onDone(result);
      try { await BackgroundService.stop(); } catch {}
    }
  };
  await BackgroundService.start(task, FG_OPTIONS);
}

export async function stopForegroundUpload() {
  try { await BackgroundService.stop(); } catch {}
}
