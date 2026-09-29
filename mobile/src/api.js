// ─────────────────────────────────────────────────────────
//  api.js — lumina-backend (web_app.py 기반 API) 호출 클라이언트
//
//  백엔드: EXPO_PUBLIC_API_URL (실제 운영 서버)
//  모든 API(조회/업로드/AI분석)가 이 서버를 가리킴
// ─────────────────────────────────────────────────────────
import { Platform } from 'react-native';
import * as FileSystem from 'expo-file-system/legacy';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { log, preview } from './logger';
import { loadLocalBundle, downloadBundle, bundleSection, bundleLifestyle, bundlePerson } from './bundle';

// ▶ 운영 백엔드 (EXPO_PUBLIC_API_URL)
export const BACKEND_URL = (process.env.EXPO_PUBLIC_API_URL || 'http://127.0.0.1:8080').replace(/\/$/, '');

// ── 멀티유저: 모든 백엔드 호출에 내 user_id 를 붙임 ──
let _userId = 'default';
export function setApiUser(uid) { _userId = (uid || 'default').trim() || 'default'; }
// 앱 시작 시 저장된 user_id 로드
export const apiReady = AsyncStorage.getItem('lumina_user_id')
  .then(v => { if (v) _userId = v; })
  .then(() => loadLocalBundle(_userId))   // 저장된 번들 있으면 로드(오프라인 우선)
  .catch(() => {});

// 번들 초기화/갱신 (UploadScreen 등에서 호출)
export const initBundle = (uid) => loadLocalBundle(uid);
export const refreshBundle = (uid) => downloadBundle(uid, BACKEND_URL);

// 번들에 값 있으면 그걸로, 없으면 온라인 호출 (오프라인 우선 + 폴백)
const fromBundleOr = (val, onlineFn) => (val !== undefined ? Promise.resolve(val) : onlineFn());

function withUser(path) {
  if (/[?&]user_id=/.test(path)) return path;   // 이미 있으면 그대로
  const sep = path.includes('?') ? '&' : '?';
  return `${path}${sep}user_id=${encodeURIComponent(_userId)}`;
}

// 공통 요청 + 로깅. 실패 시 null 반환(화면이 안 깨지게), 이유는 로그로.
async function request(method, path0, body) {
  const path = withUser(path0);
  const url = `${BACKEND_URL}${path}`;
  const t0 = Date.now();
  log.info('api', `→ ${method} ${path}${body ? ' ' + preview(body, 120) : ''}`);
  try {
    const opts = body
      ? { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
      : { method };
    const res = await fetch(url, opts);
    const ms = Date.now() - t0;
    if (!res.ok) {
      let errBody = '';
      try { errBody = await res.text(); } catch {}
      log.warn('api', `${method} ${path} → ${res.status} ${res.statusText || ''} (${ms}ms) ${preview(errBody, 150)}`);
      return null;
    }
    const json = await res.json();
    const size = Array.isArray(json) ? `${json.length}건` : (json?.results ? `${json.results.length}건` : 'ok');
    log.info('api', `✓ ${method} ${path} → ${res.status} (${ms}ms) ${size}`);
    return json;
  } catch (e) {
    const ms = Date.now() - t0;
    // 네트워크 실패는 오프라인 등 정상 상황 → 경고(노랑)로 (빨간 전체화면 방지)
    log.warn('api', `${method} ${path} 네트워크 실패 (${ms}ms): ${e.message}  [URL=${url}]`);
    return null;
  }
}

const getJSON = (path) => request('GET', path);
const postJSON = (path, body) => request('POST', path, body || {});

// ── 검색 ──
export const search = (query) => postJSON('/api/search', { query });

// ── 관계 그래프 ── (번들 우선)
export const getGraph = () => fromBundleOr(bundleSection('graph'), () => getJSON('/api/graph'));

// ── 인물 ── (번들 우선)
export const getAllProfiles = () => fromBundleOr(bundleSection('all_profiles'), () => getJSON('/api/all_profiles'));
export const getPerson = (id) => fromBundleOr(bundlePerson(id), () => getJSON(`/api/person/${id}`));
export const getPersonNames = () => fromBundleOr(bundleSection('person_names'), () => getJSON('/api/person_names'));
export const setPersonName = (person_id, name) => postJSON('/api/person_names', { person_id, name });   // 쓰기는 온라인
export const getPersonZeroRanking = () => fromBundleOr(bundleSection('person_zero_ranking'), () => getJSON('/api/person_zero_ranking'));

// ── 푸시 토큰 등록 (서버 푸시: 분석완료 알림용) ──
export const registerPushToken = (userId, token) =>
  postJSON('/api/push_token', { user_id: userId, token });

// ── 분석 상태 (폴링) ──
// 동료의 새 ai_server.py 가 제공하는 /api/ai_server/status 를 호출합니다.
export const getStatus = (userId) =>
  getJSON(`/api/ai_server/status?user_id=${encodeURIComponent(userId || 'default')}`);

// ── 라이프스타일 ── (번들 우선)
export const getLifestyleEvents = () => fromBundleOr(bundleLifestyle('events'), () => getJSON('/api/lifestyle/events'));
export const getTemporal = () => fromBundleOr(bundleLifestyle('temporal'), () => getJSON('/api/lifestyle/temporal'));
export const getYearly = () => fromBundleOr(bundleLifestyle('yearly'), () => getJSON('/api/lifestyle/yearly'));
export const getSummary = () => fromBundleOr(bundleLifestyle('summary'), () => getJSON('/api/lifestyle/summary'));
export const getLifestyleStats = () => fromBundleOr(bundleLifestyle('stats'), () => getJSON('/api/lifestyle/stats'));
// 사진별 등장인물은 파생 데이터 → 온라인 (오프라인 시 인물카드만 생략됨)
export const getPhotoPeople = (filename) =>
  getJSON(`/api/photo_people?filename=${encodeURIComponent(filename || '')}`);

// ── 이미지 URL 헬퍼 ──
// 백엔드가 주는 thumbnail_path (예: "/thumbnail/00000_name.jpg") 앞에 BASE 만 붙이면 됨
export const mediaUrl = (path) => (path ? `${BACKEND_URL}${withUser(path)}` : null);
export const thumbnailUrl = (filename) => `${BACKEND_URL}${withUser(`/thumbnail/${filename}`)}`;
export const originalUrl = (filename) => `${BACKEND_URL}${withUser(`/original/${filename}`)}`;
export const personImageUrl = (personId, filename) => `${BACKEND_URL}${withUser(`/image/${personId}/${filename}`)}`;

// ── 갤러리 업로드 ──────────────────────────────────────────
// 동료의 분석 서버 엔드포인트(/upload). 폰 갤러리 사진을 하나씩 보내면
// 동료 시스템이 user_id 폴더에 저장 → 분석 → 결과 파일 생성.
// ▶ 동료 서버 주소로 변경하세요.
export const UPLOAD_URL = `${BACKEND_URL}/upload`;
// 여러 장(640 리사이즈)을 zip 1개로 묶어 한 번에 업로드 (≤100MB)
export const UPLOAD_BATCH_URL = `${BACKEND_URL}/upload_batch`;

// zip 1개 업로드. 성공 시 서버 응답(JSON) 반환, 실패 시 throw.
export async function uploadBatchZip(zipUri, userId, count) {
  const t0 = Date.now();
  log.info('upload', `→ batch zip (${count}장, user=${userId})`);
  const result = await FileSystem.uploadAsync(UPLOAD_BATCH_URL, zipUri, {
    httpMethod: 'POST',
    uploadType: FileSystem.FileSystemUploadType.MULTIPART,
    fieldName: 'file',
    mimeType: 'application/zip',
    parameters: { user_id: userId || 'unknown', count: String(count) },
  });
  const ms = Date.now() - t0;
  if (result.status < 200 || result.status >= 300) {
    log.warn('upload', `✖ batch → ${result.status} (${ms}ms) ${preview(result.body, 150)}`);
    throw new Error(`HTTP ${result.status} ${preview(result.body, 120)}`);
  }
  let json = {};
  try { json = JSON.parse(result.body); } catch {}
  log.info('upload', `✓ batch ${count}장 → ${result.status} (${ms}ms) saved=${json.saved ?? '?'}`);
  return json;
}

function mimeFor(filename) {
  const n = (filename || '').toLowerCase();
  if (n.endsWith('.png')) return 'image/png';
  if (n.endsWith('.webp')) return 'image/webp';
  if (n.endsWith('.heic')) return 'image/heic';
  if (n.endsWith('.gif')) return 'image/gif';
  if (n.endsWith('.mp4')) return 'video/mp4';
  return 'image/jpeg';
}

// ── AI 분석 서버 (동료) ────────────────────────────────────
//  ▶ 동료가 알려준 실제 경로로 바꾸세요 (지금은 추정값).
export const AI_SERVER = BACKEND_URL;
export const AI_START_PATH = '/api/ai_server/start';   // 분석 시작 API
export const AI_PROGRESS_PATH = '/api/ai_server/status';  // 새 관제 서버의 상태 API

// 분석 시작 — AI 서버 가동
export async function startAnalysis(userId) {
  const url = `${AI_SERVER}${AI_START_PATH}`;
  log.info('ai', `→ 분석시작 ${url} (user=${userId})`);
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: userId }),
  });
  const body = await res.text();
  log.info('ai', `분석시작 ${res.status} ${preview(body, 150)}`);
  if (!res.ok) throw new Error(`HTTP ${res.status} ${preview(body, 100)}`);
  try { return JSON.parse(body); } catch { return { ok: true }; }
}

// 진행도 체크 — ai_server.py 의 /api/ai_server/status 응답 파싱
export async function checkAnalysisProgress(userId) {
  const url = `${AI_SERVER}${AI_PROGRESS_PATH}?user_id=${encodeURIComponent(userId)}`;
  log.info('ai', `→ 진행도 ${url}`);
  try {
    const res = await fetch(url);
    const body = await res.text();
    log.info('ai', `진행도 ${res.status} ${preview(body, 150)}`);
    let data = {};
    try { data = JSON.parse(body); } catch {}

    let progress = 0;
    if (data.status === 'completed') progress = 100;
    else if (typeof data.progress === 'number') progress = data.progress;

    return { progress: Number.isFinite(progress) ? progress : 0, raw: data };
  } catch (e) {
    log.error('ai', `진행도 실패: ${e.message}`);
    return { progress: 0, error: e.message };
  }
}

// 사진 1장 업로드. RN0.85 새아키텍처 FormData 버그를 피하려 FileSystem.uploadAsync 사용.
// (multipart 를 네이티브로 처리 → JS FormData 안 씀)
export async function uploadMedia(localUri, filename, userId, meta = {}) {
  const t0 = Date.now();
  const name = filename || `photo_${Date.now()}.jpg`;
  log.info('upload', `→ ${name} (user=${userId}, ${meta.media_type || 'image'}) uri=${localUri}`);
  // multipart 파일명을 원본명으로 만들려고, 결과를 원본명으로 복사 후 업로드
  const dest = `${FileSystem.cacheDirectory}${name}`;
  try {
    await FileSystem.copyAsync({ from: localUri, to: dest });
    const result = await FileSystem.uploadAsync(UPLOAD_URL, dest, {
      httpMethod: 'POST',
      uploadType: FileSystem.FileSystemUploadType.MULTIPART,
      fieldName: 'file',
      mimeType: mimeFor(name),
      parameters: {
        user_id: userId || 'unknown',
        // ▶ 메타데이터 보존 (백엔드가 .meta.json 으로 저장 → 모델이 사용)
        captured_at: meta.captured_at || '',
        lat: meta.lat != null ? String(meta.lat) : '',
        lon: meta.lon != null ? String(meta.lon) : '',
        media_type: meta.media_type || 'image',
        // ▶ 오프라인 번들용: 폰 내부 사진 식별자 (asset_id 1순위)
        asset_id: meta.asset_id || '',
        asset_uri: meta.asset_uri || '',
        platform: meta.platform || '',
      },
    });
    const ms = Date.now() - t0;
    if (result.status < 200 || result.status >= 300) {
      log.warn('upload', `✖ ${name} → ${result.status} (${ms}ms) ${preview(result.body, 150)}`);
      throw new Error(`HTTP ${result.status} ${preview(result.body, 100)}`);
    }
    let json = {};
    try { json = JSON.parse(result.body); } catch {}
    log.info('upload', `✓ ${name} → ${result.status} (${ms}ms) saved=${json.saved_filename || ''}`);
    return json;
  } catch (e) {
    log.error('upload', `✖ ${name}: ${e.message}`);
    throw e;
  } finally {
    try { await FileSystem.deleteAsync(dest, { idempotent: true }); } catch {}
  }
}

// ── 촬영시각 예측 (단일 이미지 1장) ────────────────────────────
//  사진 1장을 서버 /api/predict_time 으로 보내면 GPU 모델서버가 촬영시각을 예측.
//  반환: { ok, predicted_time "MM-DD HH:MM", predicted_time_full, actual_time, caption }
//  실패 시 throw.
export async function predictTime(localUri, filename, userId) {
  const t0 = Date.now();
  const name = filename || `photo_${Date.now()}.jpg`;
  const dest = `${FileSystem.cacheDirectory}${name}`;
  const url = `${BACKEND_URL}/api/predict_time`;
  log.info('predict', `→ ${name} (user=${userId || _userId})`);
  try {
    await FileSystem.copyAsync({ from: localUri, to: dest });
    const result = await FileSystem.uploadAsync(url, dest, {
      httpMethod: 'POST',
      uploadType: FileSystem.FileSystemUploadType.MULTIPART,
      fieldName: 'image',                 // 서버 time_model_server 가 기대하는 필드명
      mimeType: mimeFor(name),
      parameters: { user_id: userId || _userId || 'unknown' },
    });
    const ms = Date.now() - t0;
    let json = {};
    try { json = JSON.parse(result.body); } catch {}
    if (result.status < 200 || result.status >= 300 || json.ok === false) {
      log.warn('predict', `✖ ${name} → ${result.status} (${ms}ms) ${preview(result.body, 150)}`);
      throw new Error(json.error || `HTTP ${result.status}`);
    }
    log.info('predict', `✓ ${name} → ${json.predicted_time || '?'} (${ms}ms)`);
    return json;
  } catch (e) {
    log.error('predict', `✖ ${name}: ${e.message}`);
    throw e;
  } finally {
    try { await FileSystem.deleteAsync(dest, { idempotent: true }); } catch {}
  }
}

// ── 회원가입 / 로그인 ──────────────────────────────────────
const TOKEN_KEY = 'lumina_token';

// 인증 요청: 실패(401/409 등)도 에러 메시지를 살려 반환 (request() 와 달리 null 아님)
async function authRequest(path, body) {
  const url = `${BACKEND_URL}${path}`;
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    log.info('auth', `${path} → ${res.status}`);
    return { ok: res.ok, status: res.status, data };
  } catch (e) {
    log.warn('auth', `${path} 네트워크 실패: ${e.message}`);
    return { ok: false, status: 0, data: { error: '네트워크 오류: ' + e.message } };
  }
}

export const register = (userId, password) => authRequest('/api/register', { user_id: userId, password });
export const login = (userId, password) => authRequest('/api/login', { user_id: userId, password });
export const logout = (userId) => authRequest('/api/logout', { user_id: userId });

// 로그인 성공 후 세션 저장 (UID_KEY 재사용 + 토큰) / 해제 / 조회
export async function saveSession(userId, token) {
  await AsyncStorage.setItem('lumina_user_id', userId);
  if (token) await AsyncStorage.setItem(TOKEN_KEY, token);
  setApiUser(userId);
}
export async function clearSession() {
  await AsyncStorage.multiRemove(['lumina_user_id', TOKEN_KEY]);
  setApiUser('default');
}
export const getSavedUser = () => AsyncStorage.getItem('lumina_user_id');

// ── 업로드된 사진 목록 / 삭제 (분석 전 관리) ──────────────────
// getUploads: { count, files: [{ saved_filename, original_filename, asset_id, captured_at, size }] }
export const getUploads = (userId) =>
  getJSON(`/api/uploads?user_id=${encodeURIComponent(userId || _userId)}`);

// deleteUpload: 서버 uploads 에서 해당 사진(+메타) 삭제. 폰 원본은 유지.
export async function deleteUpload(userId, savedFilename) {
  const url = `${BACKEND_URL}/api/uploads?user_id=${encodeURIComponent(userId || _userId)}&filename=${encodeURIComponent(savedFilename)}`;
  log.info('uploads', `→ DELETE ${savedFilename}`);
  try {
    const res = await fetch(url, { method: 'DELETE' });
    const data = await res.json().catch(() => ({}));
    return { ok: res.ok && data.ok !== false, data };
  } catch (e) {
    log.warn('uploads', `삭제 실패: ${e.message}`);
    return { ok: false, data: { error: e.message } };
  }
}

// deleteAllUploads: 서버 uploads 의 해당 유저 전체 삭제 (폰 원본 유지).
export async function deleteAllUploads(userId) {
  const url = `${BACKEND_URL}/api/uploads?user_id=${encodeURIComponent(userId || _userId)}&all=1`;
  log.info('uploads', `→ DELETE ALL (user=${userId || _userId})`);
  try {
    const res = await fetch(url, { method: 'DELETE' });
    const data = await res.json().catch(() => ({}));
    return { ok: res.ok && data.ok !== false, data };
  } catch (e) {
    log.warn('uploads', `전체 삭제 실패: ${e.message}`);
    return { ok: false, data: { error: e.message } };
  }
}
