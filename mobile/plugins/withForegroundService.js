// react-native-background-actions 의 포그라운드 서비스에 foregroundServiceType="dataSync" 를
// 병합해 줌 (Android 14+ 에서 필수, 없으면 startForeground 시 크래시).
const { withAndroidManifest } = require('expo/config-plugins');

const SERVICE = 'com.asterinet.react.bgactions.RNBackgroundActionsTask';

module.exports = function withForegroundService(config) {
  return withAndroidManifest(config, (cfg) => {
    const manifest = cfg.modResults.manifest;
    manifest.$['xmlns:tools'] = 'http://schemas.android.com/tools';
    const app = manifest.application && manifest.application[0];
    if (!app) return cfg;
    app.service = app.service || [];
    if (!app.service.some((s) => s.$ && s.$['android:name'] === SERVICE)) {
      app.service.push({
        $: {
          'android:name': SERVICE,
          'android:foregroundServiceType': 'dataSync',
          'tools:node': 'merge',
        },
      });
    }
    return cfg;
  });
};
