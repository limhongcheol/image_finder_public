// metro.config.js
// Windows 에서 @msgpackr-extract 등 타 플랫폼 바이너리 폴더를 Metro 가 감시하다
// "ENOENT watch ..." 로 죽는 문제 방지 → 해당 폴더를 감시/해석 대상에서 제외.
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

const prev = config.resolver.blockList;
const prevList = Array.isArray(prev) ? prev : prev ? [prev] : [];
config.resolver.blockList = [
  ...prevList,
  /node_modules[\\/]@msgpackr-extract[\\/].*/,
];

module.exports = config;
