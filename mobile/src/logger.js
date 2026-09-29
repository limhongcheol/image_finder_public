// ─────────────────────────────────────────────────────────
//  logger.js — 통일된 콘솔 로깅
//  로그는 Metro 터미널(npx expo start 창) + 폰 dev 메뉴에 보임.
//  사용: import { log } from './logger';  log.info('api', '...')
// ─────────────────────────────────────────────────────────
const time = () => {
  const d = new Date();
  const p = (n, l = 2) => String(n).padStart(l, '0');
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}.${p(d.getMilliseconds(), 3)}`;
};

const fmt = (tag, mark) => `${time()} ${mark}[${tag}]`;

export const log = {
  info:  (tag, ...a) => console.log(fmt(tag, ''), ...a),
  warn:  (tag, ...a) => console.warn(fmt(tag, '⚠ '), ...a),
  error: (tag, ...a) => console.error(fmt(tag, '✖ '), ...a),
};

// 본문 미리보기(너무 길면 자름)
export const preview = (v, n = 200) => {
  try {
    const s = typeof v === 'string' ? v : JSON.stringify(v);
    return s.length > n ? s.slice(0, n) + `…(+${s.length - n})` : s;
  } catch {
    return String(v);
  }
};
