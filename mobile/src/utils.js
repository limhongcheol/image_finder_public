export function formatSize(bytes) {
  if (!bytes) return '—';
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

// expo-media-library 에셋을 분류
export function classifyAsset(asset) {
  if (asset.mediaType === 'video') return 'video';
  if (asset.mediaType === 'photo') return 'image';
  return null;
}

// 에셋은 이미 width/height를 포함 (getAssetInfoAsync 후)
export function getAssetDimensions(asset) {
  return { w: asset.width || 0, h: asset.height || 0 };
}
