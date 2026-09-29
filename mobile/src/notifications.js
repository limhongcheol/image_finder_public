// ─────────────────────────────────────────────────────────
//  notifications.js — 로컬 푸시 알림 (업로드 완료 / AI 분석 완료)
//  ⚠️ 앱이 완전히 종료된 상태의 '분석 완료'는 서버 푸시(Expo Push)가 필요. 여기선 로컬 알림.
// ─────────────────────────────────────────────────────────
import * as Notifications from 'expo-notifications';
import { Platform } from 'react-native';
import { log } from './logger';

// 앱이 떠 있을 때도 배너로 보이게
Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowBanner: true,
    shouldShowList: true,
    shouldPlaySound: true,
    shouldSetBadge: false,
  }),
});

let _granted = false;

export async function ensureNotificationPermission() {
  try {
    if (Platform.OS === 'android') {
      await Notifications.setNotificationChannelAsync('lumina', {
        name: 'Lumina',
        importance: Notifications.AndroidImportance.DEFAULT,
      });
    }
    const cur = await Notifications.getPermissionsAsync();
    let status = cur.status;
    if (status !== 'granted') {
      const req = await Notifications.requestPermissionsAsync();
      status = req.status;
    }
    _granted = status === 'granted';
    log.info('notif', `권한 ${status}`);
    return _granted;
  } catch (e) {
    log.error('notif', '권한 실패: ' + e.message);
    return false;
  }
}

// Expo 푸시 토큰 발급 (서버 푸시용). 실패 시 null (FCM 미설정/에뮬레이터 등).
const PROJECT_ID = process.env.EXPO_PUBLIC_EAS_PROJECT_ID || '';
export async function getPushToken() {
  try {
    if (!PROJECT_ID) return null;
    const granted = await ensureNotificationPermission();
    if (!granted) { log.warn('notif', '알림 권한 없음 → 푸시 토큰 스킵'); return null; }
    const res = await Notifications.getExpoPushTokenAsync({ projectId: PROJECT_ID });
    log.info('notif', '푸시 토큰 발급 완료');
    return res.data;
  } catch (e) {
    // 오프라인/FCM 미설정/에뮬레이터 등 → 정상 상황이라 경고(노랑)로 (빨간 전체화면 방지)
    log.warn('notif', '푸시 토큰 실패 (오프라인/FCM 미설정/실기기 아님?): ' + e.message);
    return null;
  }
}

// 즉시 로컬 알림
export async function notify(title, body, data = {}) {
  try {
    await Notifications.scheduleNotificationAsync({
      content: { title, body, data },
      trigger: null,
    });
    log.info('notif', `알림: ${title} — ${body}`);
  } catch (e) {
    log.error('notif', '알림 실패: ' + e.message);
  }
}
