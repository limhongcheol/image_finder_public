// 서버 도달 가능(온라인) 여부 — 네이티브 모듈 없이 짧은 fetch.
// 공유 싱글톤: 여러 컴포넌트(FaceImage 등 다수)가 핑 1번 결과를 같이 씀.
import { useEffect, useState } from 'react';
import { BACKEND_URL } from './api';

export async function pingOnline() {
  try {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 4000);
    const res = await fetch(`${BACKEND_URL}/`, { method: 'GET', signal: ctrl.signal });
    clearTimeout(t);
    return res.status > 0;
  } catch {
    return false;
  }
}

let _online = null;          // null=확인중, true, false
const subs = new Set();
let _timer = null;

async function _refresh() {
  const v = await pingOnline();
  if (v !== _online) { _online = v; subs.forEach(fn => fn(v)); }
}

// null=확인중, true=온라인, false=오프라인 (공유 상태)
export function useOnline() {
  const [online, setOnline] = useState(_online);
  useEffect(() => {
    subs.add(setOnline);
    setOnline(_online);
    if (subs.size === 1) {           // 첫 구독자만 핑/주기검사 시작
      _refresh();
      _timer = setInterval(_refresh, 15000);
    }
    return () => {
      subs.delete(setOnline);
      if (subs.size === 0 && _timer) { clearInterval(_timer); _timer = null; }
    };
  }, []);
  return online;
}
