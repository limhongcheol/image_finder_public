// ─────────────────────────────────────────────────────────
//  gallery.js — 폰 갤러리 인덱스
//
//  분석 데이터는 사진을 filename 으로 식별(예: IMG_2014..._161636.jpg).
//  폰 갤러리 자산도 filename 을 가지므로, filename → 폰 사진 uri 맵을 만들어
//  "원본은 폰에서" 표시. 폰에 없으면 호출부에서 백엔드로 폴백.
//
//  얼굴 crop 등 파생 이미지는 폰에 없으므로 여기서 다루지 않음(백엔드 사용).
// ─────────────────────────────────────────────────────────
import { createContext, useCallback, useContext, useEffect, useState } from 'react';
import * as MediaLibrary from 'expo-media-library/legacy';

let _index = null;   // Map: filename(소문자) → 폰 자산 uri  (모듈 캐시, 1회만 구축)

const GalleryCtx = createContext({ ready: false, status: 'idle', localUriFor: () => null, allPhotos: () => [], assetIdFor: () => null });
export const useGallery = () => useContext(GalleryCtx);

export function GalleryProvider({ children }) {
  const [ready, setReady] = useState(_index != null);
  const [status, setStatus] = useState(_index != null ? 'ready' : 'idle');

  useEffect(() => {
    if (_index != null) { setReady(true); return; }
    (async () => {
      try {
        const perm = await MediaLibrary.requestPermissionsAsync();
        if (!perm.granted) { setStatus('denied'); return; }
        setStatus('indexing');
        const idx = new Map();
        let after;
        let hasNext = true;
        while (hasNext) {
          const page = await MediaLibrary.getAssetsAsync({
            mediaType: [MediaLibrary.MediaType.photo, MediaLibrary.MediaType.video],
            first: 1000,
            after,
          });
          for (const a of page.assets) {
            if (a.filename) idx.set(a.filename.toLowerCase(), { uri: a.uri, id: a.id });
          }
          after = page.endCursor;
          hasNext = page.hasNextPage;

          // 수만 장 스캔 시 앱이 멈추고 튕기는 현상(ANR) 방지를 위해 아주 짧게 쉼
          await new Promise(r => setTimeout(r, 10));
        }
        _index = idx;
        setStatus('ready');
        setReady(true);
      } catch (e) {
        console.warn('[gallery] 인덱싱 실패:', e.message);
        setStatus('error');
      }
    })();
  }, []);

  // ready 가 바뀌면 새 함수 → 화면 리렌더 (인덱스 준비 후 폰 사진으로 갱신)
  const localUriFor = useCallback((filename) => {
    if (!_index || !filename) return null;
    const e = _index.get(String(filename).toLowerCase());
    return e ? e.uri : null;
  }, [ready]);

  // 갤러리 전체 [{ filename(소문자), uri, id }] — 미업로드 그리드용 (이미 만든 _index 재사용)
  const allPhotos = useCallback(() => {
    if (!_index) return [];
    return Array.from(_index, ([filename, e]) => ({ filename, uri: e.uri, id: e.id }));
  }, [ready]);

  // 파일명 → 폰 자산 id (삭제 후 upKey 정리용)
  const assetIdFor = useCallback((filename) => {
    if (!_index || !filename) return null;
    const e = _index.get(String(filename).toLowerCase());
    return e ? e.id : null;
  }, [ready]);

  return (
    <GalleryCtx.Provider value={{ ready, status, localUriFor, allPhotos, assetIdFor }}>
      {children}
    </GalleryCtx.Provider>
  );
}
