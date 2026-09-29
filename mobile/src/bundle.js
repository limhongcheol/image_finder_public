// ─────────────────────────────────────────────────────────
//  bundle.js — 오프라인 번들 (GET /api/bundle 의 JSON 1개)
//
//  분석 완료 후 서버가 모든 결과를 JSON 1개로 내려줌 → 폰에 저장 → 오프라인 렌더.
//  번들 없으면(미배포/구버전/실패) 전부 undefined 반환 → api.js 가 온라인 폴백.
//  사진은 photos[] 의 local_ref(asset_id) 로 폰 갤러리에서 직접 로드.
// ─────────────────────────────────────────────────────────
import * as FileSystem from 'expo-file-system/legacy';
import { log } from './logger';

const SCHEMA_VERSION = 1;   // 앱이 이해하는 번들 버전 (상위면 온라인 폴백)
const bundlePath = (uid) => `${FileSystem.documentDirectory}bundle_${encodeURIComponent(uid)}.json`;

let _bundle = null;   // 현재 로드된 번들 (메모리)
let _user = null;

export function getBundle() { return _bundle; }
export function hasBundle() { return !!_bundle; }

// 로컬 저장 번들 로드 (앱 시작 / user_id 설정 시)
export async function loadLocalBundle(uid) {
  _user = uid;
  if (!uid) { _bundle = null; return null; }
  try {
    const path = bundlePath(uid);
    const info = await FileSystem.getInfoAsync(path);
    if (!info.exists) { _bundle = null; return null; }
    const b = JSON.parse(await FileSystem.readAsStringAsync(path));
    if ((b.schema_version || 1) > SCHEMA_VERSION) {
      log.warn('bundle', `schema ${b.schema_version} > ${SCHEMA_VERSION} → 온라인 폴백`);
      _bundle = null; return null;
    }
    _bundle = b;
    log.info('bundle', `로컬 번들 로드 ✓ (generated_at ${b.generated_at || '?'})`);
    return b;
  } catch (e) {
    log.error('bundle', '로컬 번들 로드 실패: ' + e.message);
    _bundle = null; return null;
  }
}

// 서버에서 번들 다운로드 → 로컬 저장 + 로드. 실패/404 면 기존 유지.
export async function downloadBundle(uid, baseUrl) {
  if (!uid) return null;
  const url = `${baseUrl}/api/bundle?user_id=${encodeURIComponent(uid)}`;
  const path = bundlePath(uid);
  const tmp = path + '.tmp';
  try {
    log.info('bundle', `→ 다운로드 ${url}`);
    const res = await FileSystem.downloadAsync(url, tmp);
    if (res.status !== 200) {
      log.warn('bundle', `다운로드 ${res.status} → 번들 없음(온라인 유지)`);
      try { await FileSystem.deleteAsync(tmp, { idempotent: true }); } catch {}
      return null;
    }
    // 검증 후 교체 (깨진 응답으로 기존 번들 날리지 않게)
    const b = JSON.parse(await FileSystem.readAsStringAsync(tmp));
    await FileSystem.deleteAsync(path, { idempotent: true });
    await FileSystem.moveAsync({ from: tmp, to: path });
    log.info('bundle', `다운로드 ✓ (사진 ${b.photos?.length ?? 0}장)`);
    return await loadLocalBundle(uid);
  } catch (e) {
    log.error('bundle', '다운로드 실패: ' + e.message);
    try { await FileSystem.deleteAsync(tmp, { idempotent: true }); } catch {}
    return null;
  }
}

// ── 섹션 접근자 (없으면 undefined → 호출측 온라인 폴백) ──
export function bundleSection(name) {
  return _bundle && _bundle.sections ? _bundle.sections[name] : undefined;
}
export function bundleLifestyle(name) {
  const ls = _bundle && _bundle.sections && _bundle.sections.lifestyle;
  return ls ? ls[name] : undefined;
}
export function bundlePerson(id) {
  const ps = _bundle && _bundle.sections && _bundle.sections.persons;
  return ps ? ps[id] : undefined;
}

// 사진 룩업: filename 또는 saved_filename 으로 photos[] 항목 (local_ref 포함) 반환
export function bundlePhotoByFilename(filename) {
  if (!_bundle || !_bundle.photos || !filename) return null;
  const base = String(filename).replace(/.*[\\/]/, '');
  return _bundle.photos.find(
    p => p.filename === filename || p.saved_filename === filename
      || p.filename === base || p.saved_filename === base
  ) || null;
}
