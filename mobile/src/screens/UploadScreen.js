import { useEffect, useRef, useState, useCallback } from 'react';
import {
  View, Text, TextInput, TouchableOpacity, ActivityIndicator, StyleSheet, Alert, ScrollView,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { getStatus, startAnalysis, registerPushToken, refreshBundle, UPLOAD_URL, getUploads } from '../api';
import {
  syncNewMedia, syncNewMediaBatch, runForegroundUpload, stopForegroundUpload,
} from '../uploadSync';

// ▶ /upload_batch 배포 확인됨(405) → 배치 사용.
const USE_BATCH_UPLOAD = true;

// ▶ AI 분석 소요시간 안내 (실측: 1000장당 약 1시간 = 1장당 ~4초)
const EST_NOTE = '분석은 사진 약 1000장당 1시간 정도 소요돼요 (1장당 ~4초). 사진이 많으면 그만큼 더 걸리고, 완료되면 알림으로 알려드려요.';
import { ensureNotificationPermission, notify, getPushToken } from '../notifications';
import { useOnline } from '../useOnline';
import OfflineNotice from '../components/OfflineNotice';
import UploadGalleryModal from '../components/UploadGalleryModal';
import ConsentModal, { consentKey } from '../components/ConsentModal';
import { ACCENT, BLUE, BG, TEXT, MUTED, PADDING } from '../theme';

export default function UploadScreen({ userId: propUserId, onLogout }) {
  const userId = (propUserId || '').trim();     // 로그인된 아이디 (App.js 에서 전달)
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState({ done: 0, total: 0, ok: 0, fail: 0, current: '' });
  const [lastResult, setLastResult] = useState(null);   // 마지막 업로드 결과 (완료 후 화면 표시)
  const [uploadCount, setUploadCount] = useState(null); // 서버에 올라간 사진 수
  const [showGallery, setShowGallery] = useState(false);// 업로드 사진 관리 모달
  const [status, setStatus] = useState(null);   // 분석 상태 (폴링)
  // ── AI 분석 흐름 ──
  const [aiStarted, setAiStarted] = useState(false);   // 분석 시작됨 → 사진 추가 잠금
  const [aiBusy, setAiBusy] = useState(false);
  const cancelRef = useRef(false);
  const notifiedDoneRef = useRef(false);   // 분석완료 알림 중복 방지
  const bundleFetchedRef = useRef(false);  // 세션당 번들 1회 다운로드
  // ── 개인정보 수집·이용 동의 (Play 스토어 Prominent Disclosure) ──
  const [consented, setConsented] = useState(false);   // 동의 완료 여부
  const [showConsent, setShowConsent] = useState(false);
  const pendingUploadRef = useRef(null);   // 동의 후 이어서 실행할 업로드 인자

  const aiKey = (uid) => `lumina_ai_started_${uid}`;
  const aiNotifKey = (uid) => `lumina_ai_notified_${uid}`;   // 완료 알림 1회 기록

  useEffect(() => {
    ensureNotificationPermission();   // 푸시 알림 권한
  }, []);

  // 서버에 올라간 사진 수 로드 (화면 진입 / 유저 변경 / 업로드 완료 시 갱신)
  const loadUploadCount = useCallback(async () => {
    if (!userId) return;
    const r = await getUploads(userId);
    setUploadCount(r?.count ?? null);
  }, [userId]);
  useEffect(() => { loadUploadCount(); }, [loadUploadCount]);

  // 분석 시작 여부 복원 (앱 재시작해도 잠금 유지)
  useEffect(() => {
    const uid = userId.trim();
    if (!uid) return;
    AsyncStorage.getItem(aiKey(uid)).then(v => setAiStarted(v === '1'));
  }, [userId]);

  // 개인정보 동의 여부 복원 (유저별 1회 동의)
  useEffect(() => {
    const uid = userId.trim();
    if (!uid) { setConsented(false); return; }
    AsyncStorage.getItem(consentKey(uid)).then(v => setConsented(v === '1'));
  }, [userId]);

  // 푸시 토큰 발급 → 서버 등록 (앱 닫혀도 분석완료 알림 받기)
  useEffect(() => {
    const uid = userId.trim();
    if (!uid) return;
    (async () => {
      const token = await getPushToken();
      if (token) await registerPushToken(uid, token);
    })();
  }, [userId]);

  const doStartAnalysis = async () => {
    const uid = userId.trim();
    if (!uid) { Alert.alert('user_id 필요', '먼저 식별 ID를 입력하세요.'); return; }
    Alert.alert('AI 분석 시작', '분석을 시작하면 더 이상 사진을 추가할 수 없습니다. 계속할까요?', [
      { text: '취소', style: 'cancel' },
      {
        text: '시작', style: 'destructive',
        onPress: async () => {
          setAiBusy(true);
          try {
            await startAnalysis(uid);
            setAiStarted(true);
            await AsyncStorage.setItem(aiKey(uid), '1');
            await AsyncStorage.removeItem(aiNotifKey(uid));   // 새 분석 → 완료 시 다시 알림 가능
            notifiedDoneRef.current = false;
            Alert.alert('분석 시작됨', '이제 사진 추가는 잠깁니다. 진행도를 확인하세요.');
          } catch (e) {
            Alert.alert('분석 시작 실패', String(e?.message || e));
          } finally { setAiBusy(false); }
        },
      },
    ]);
  };

  // 분석 상태 폴링 + 완료 알림 (앱 재실행해도 딱 한 번만 — AsyncStorage 영구 기록)
  useEffect(() => {
    const uid = userId.trim();
    if (!uid) return;
    let alive = true;
    let id;
    bundleFetchedRef.current = false;   // 유저 바뀌면 번들 다시 받기
    const notifKey = aiNotifKey(uid);
    const tick = async () => {
      const s = await getStatus(uid);
      if (!alive || !s) return;
      setStatus(s);
      if (s.status === 'completed') {
        if (!notifiedDoneRef.current) {                 // 완료 알림 기록(서버 푸시가 보냄)
          notifiedDoneRef.current = true;
          await AsyncStorage.setItem(notifKey, '1');
        }
        if (!bundleFetchedRef.current) {                // 완료 → 오프라인 번들 다운로드 (세션 1회)
          bundleFetchedRef.current = true;
          refreshBundle(uid);
        }
      } else if (notifiedDoneRef.current) {              // 새 분석 사이클 → 기록 초기화
        notifiedDoneRef.current = false;
        await AsyncStorage.removeItem(notifKey);
      }
    };
    (async () => {
      // 시작 전: 이미 알림 보낸 적 있으면 ref=true (앱 열 때마다 울리는 버그 방지)
      const v = await AsyncStorage.getItem(notifKey);
      if (!alive) return;
      notifiedDoneRef.current = (v === '1');
      await tick();
      id = setInterval(tick, 1000);   // 1초마다 자동 갱신
    })();
    return () => { alive = false; if (id) clearInterval(id); };
  }, [userId]);

  const stop = () => { cancelRef.current = true; stopForegroundUpload(); };

  // 업로드 시작 진입점: 최초 1회 개인정보 동의를 받은 뒤에만 실제 업로드 실행
  const start = (limit = 0) => {
    const uid = userId.trim();
    if (!uid) { Alert.alert('user_id 필요', '먼저 본인 식별 ID를 입력하세요.'); return; }
    if (!consented) {
      pendingUploadRef.current = limit;   // 동의 후 이어서 실행
      setShowConsent(true);
      return;
    }
    runUpload(limit);
  };

  const onConsentAgree = async () => {
    const uid = userId.trim();
    setShowConsent(false);
    setConsented(true);
    await AsyncStorage.setItem(consentKey(uid), '1');
    const limit = pendingUploadRef.current;
    pendingUploadRef.current = null;
    if (limit !== null && limit !== undefined) runUpload(limit);
  };

  const onConsentDecline = () => {
    setShowConsent(false);
    pendingUploadRef.current = null;
  };

  const runUpload = async (limit = 0) => {
    const uid = userId.trim();
    cancelRef.current = false;
    setRunning(true);
    setProgress({ done: 0, total: 0, ok: 0, fail: 0, current: '' });

    const sync = USE_BATCH_UPLOAD ? syncNewMediaBatch : syncNewMedia;
    const onProgress = (p) => setProgress(prev => ({ ...prev, ...p }));
    const onDone = (r) => {
      setRunning(false);
      if (!r.cancelled) {
        notify('업로드 완료 ✅', `사진 ${r.ok}장 전송 완료${r.fail > 0 ? `, 실패 ${r.fail}장` : ''}`);
      }
      Alert.alert('업로드 종료',
        `성공 ${r.ok} · 실패 ${r.fail}${r.cancelled ? ' (중단됨)' : ''}` +
        (r.fail > 0 && r.lastErr ? `\n\n첫 에러: ${r.lastErr}` : ''));
      setLastResult(r);   // 완료 후 화면에 남김
      loadUploadCount();  // 업로드 후 현황 갱신
    };

    try {
      // 포그라운드 서비스로 실행 → 앱 닫아도 계속 전송 ("업로드 중" 상시 알림)
      await runForegroundUpload(uid, sync, { limit, onProgress, onDone });
    } catch (e) {
      // 포그라운드 서비스 불가 → 일반 실행으로 폴백
      try {
        const r = await sync(uid, { limit, onProgress, isCancelled: () => cancelRef.current });
        onDone(r);
      } catch (e2) {
        setRunning(false);
        Alert.alert('업로드 실패', String(e2?.message || e2));
      }
    }
  };

  const pct = progress.total > 0 ? Math.round(progress.done / progress.total * 100) : 0;
  const online = useOnline();
  // 분석이 진행 중인지 / 중단(서버 재시작 등)되어 재시작이 필요한지 판정
  const aiInProgress = !!status && ['queued', 'pulling', 'running', 'pushing'].includes(status.status);
  const canRestartAnalysis = aiStarted && !!status && !aiInProgress && status.status !== 'completed';

  // 업로드·AI분석은 서버 필요 → 오프라인이면 안내 화면
  if (online === false) {
    return (
      <OfflineNotice
        title="오프라인 상태예요"
        message="사진 업로드와 AI 분석은 인터넷 연결이 필요해요. 연결되면 다시 시도해주세요."
      />
    );
  }

  return (
    <>
    <ScrollView style={styles.wrap} contentContainerStyle={{ paddingBottom: 40 }}>
      <Text style={styles.title}>업로드</Text>
      <Text style={styles.sub}>갤러리 사진을 분석 서버로 전송합니다</Text>

      <Text style={styles.label}>로그인 계정</Text>
      <View style={styles.idLocked}>
        <Text style={styles.idLockedTxt}>{userId}</Text>
        <TouchableOpacity onPress={onLogout} hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}>
          <Text style={styles.logoutTxt}>로그아웃</Text>
        </TouchableOpacity>
      </View>
      <Text style={styles.endpoint}>→ {UPLOAD_URL}</Text>

      <TouchableOpacity style={styles.uploadStat} onPress={() => setShowGallery(true)} activeOpacity={0.8}>
        <Text style={styles.uploadStatTxt}>📷 업로드된 사진 {uploadCount != null ? `${uploadCount}장` : '…'}</Text>
        <Text style={styles.uploadStatArrow}>관리 ›</Text>
      </TouchableOpacity>

      {aiStarted ? (
        <View style={styles.lockedCard}>
          <Text style={styles.lockedTitle}>🔒 분석 시작됨 — 사진 추가 잠김</Text>
          <Text style={styles.lockedSub}>분석 중에는 사진을 더 추가할 수 없습니다. 아래에서 진행도를 확인하세요.</Text>
        </View>
      ) : running ? (
        <>
          <View style={styles.progressCard}>
            <Text style={styles.pct}>{pct}%</Text>
            <View style={styles.track}><View style={[styles.fill, { width: `${pct}%` }]} /></View>
            <Text style={styles.counts}>{progress.done} / {progress.total}  (성공 {progress.ok} · 실패 {progress.fail})</Text>
            <Text style={styles.current} numberOfLines={1}>{progress.current}</Text>
            <ActivityIndicator color={ACCENT} style={{ marginTop: 10 }} />
          </View>
          <TouchableOpacity style={[styles.btn, { backgroundColor: '#999' }]} onPress={stop}>
            <Text style={styles.btnTxt}>중단</Text>
          </TouchableOpacity>
        </>
      ) : (
        <>
          {lastResult && (
            <View style={styles.uploadResult}>
              <Text style={styles.uploadResultTxt}>
                {lastResult.cancelled ? '⏹ 업로드 중단됨' : '✅ 업로드 완료'}
                {'  ·  '}성공 {lastResult.ok}장{lastResult.fail > 0 ? `  ·  실패 ${lastResult.fail}장` : ''}
              </Text>
              {lastResult.fail > 0 && !!lastResult.lastErr && (
                <Text style={styles.uploadResultErr} numberOfLines={2}>첫 에러: {lastResult.lastErr}</Text>
              )}
            </View>
          )}
          <TouchableOpacity style={styles.btn} onPress={() => start(0)}>
            <Text style={styles.btnTxt}>{lastResult ? '사진 더 올리기' : '1. 사전 준비'}</Text>
          </TouchableOpacity>
        </>
      )}

      {/* AI 분석 + 서버 상태 */}
      <View style={styles.aiCard}>
        <Text style={styles.aiTitle}>🤖 AI 분석</Text>
        <Text style={styles.aiEst}>⏱ {EST_NOTE}</Text>

        {canRestartAnalysis ? (
          <TouchableOpacity
            style={[styles.btn, aiBusy && styles.btnDisabled]}
            disabled={aiBusy}
            onPress={doStartAnalysis}
          >
            <Text style={styles.btnTxt}>🔄 분석 재시작 (중단됨)</Text>
          </TouchableOpacity>
        ) : (
          <TouchableOpacity
            style={[styles.btn, (aiStarted || aiBusy) && styles.btnDisabled]}
            disabled={aiStarted || aiBusy}
            onPress={doStartAnalysis}
          >
            <Text style={styles.btnTxt}>2. 분석 시작</Text>
          </TouchableOpacity>
        )}
        {aiBusy && <ActivityIndicator color={ACCENT} style={{ marginTop: 10 }} />}

        {/* AI 서버 상태 (단계 + 퍼센트) */}
        {aiStarted && (
          <View style={styles.aiStatusBox}>
            <Text style={styles.aiStatusLabel}>AI 서버 상태</Text>
            <Text style={[styles.aiStatusTitle, status?.status === 'completed' && { color: '#3a9e5c' }]}>
              {status?.status === 'completed'
                ? '✅ 분석 완료 — 데이터 준비됨'
                : status?.status
                  ? `⏳ ${status.status}${status.progress > 0 ? `  ·  ${status.progress}%` : ''}`
                  : '⏳ 대기 중… (잠시 후 갱신)'}
            </Text>
            {!!(status && (status.message || status.queue_position)) && (
              <Text style={styles.aiStatusMeta}>
                {status.message || ''}
                {status.queue_position ? `  ·  대기 순번 ${status.queue_position}` : ''}
              </Text>
            )}
          </View>
        )}
      </View>

      <View style={styles.note}>
        <Text style={styles.noteTxt}>• 이미 올린 사진은 자동으로 건너뜁니다 (중복 방지)</Text>
        <Text style={styles.noteTxt}>• 중단 후 다시 누르면 안 올라간 것부터 이어서</Text>
        <Text style={styles.noteTxt}>• 사진이 많으면 시간이 걸립니다 (Wi-Fi 권장)</Text>
      </View>
    </ScrollView>
    <UploadGalleryModal
      visible={showGallery}
      onClose={() => setShowGallery(false)}
      userId={userId}
      onChanged={loadUploadCount}
    />
    <ConsentModal
      visible={showConsent}
      onAgree={onConsentAgree}
      onDecline={onConsentDecline}
    />
    </>
  );
}

const styles = StyleSheet.create({
  wrap: { flex: 1, backgroundColor: BG, paddingHorizontal: PADDING },
  title: { fontSize: 24, fontWeight: '700', color: TEXT, marginTop: 8 },
  sub: { fontSize: 13, color: MUTED, marginTop: 4, marginBottom: 18 },
  label: { fontSize: 13, color: TEXT, fontWeight: '600', marginBottom: 6 },
  input: {
    height: 48, backgroundColor: '#fff', borderRadius: 12, paddingHorizontal: 14,
    fontSize: 15, color: TEXT, borderWidth: 1, borderColor: 'rgba(0,0,0,0.08)',
  },
  idLocked: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', height: 48, backgroundColor: 'rgba(0,0,0,0.04)', borderRadius: 12, paddingHorizontal: 14, borderWidth: 1, borderColor: 'rgba(0,0,0,0.1)' },
  idLockedTxt: { fontSize: 15, fontWeight: '700', color: TEXT },
  idLockedTag: { fontSize: 12, color: MUTED },
  logoutTxt: { fontSize: 13, color: ACCENT, fontWeight: '600' },
  endpoint: { fontSize: 11, color: MUTED, marginTop: 6, marginBottom: 14 },
  btn: { backgroundColor: ACCENT, borderRadius: 12, height: 50, justifyContent: 'center', alignItems: 'center', marginTop: 10 },
  btnDisabled: { opacity: 0.4 },
  lockedCard: { backgroundColor: 'rgba(0,0,0,0.04)', borderRadius: 12, padding: 16, marginTop: 10, borderWidth: 1, borderColor: 'rgba(0,0,0,0.1)' },
  lockedTitle: { fontSize: 14, fontWeight: '700', color: TEXT },
  lockedSub: { fontSize: 12, color: MUTED, marginTop: 4, lineHeight: 17 },
  aiCard: { backgroundColor: '#fff', borderRadius: 12, padding: 14, marginTop: 16, borderWidth: 1, borderColor: 'rgba(0,0,0,0.08)' },
  aiTitle: { fontSize: 15, fontWeight: '700', color: TEXT, marginBottom: 6 },
  aiEst: { fontSize: 12, color: MUTED, lineHeight: 17, marginBottom: 8, backgroundColor: 'rgba(200,90,48,0.06)', padding: 10, borderRadius: 8 },
  aiStatusBox: { marginTop: 14, padding: 12, backgroundColor: 'rgba(0,0,0,0.03)', borderRadius: 10, borderWidth: 1, borderColor: 'rgba(0,0,0,0.06)' },
  aiStatusLabel: { fontSize: 11, color: MUTED, fontWeight: '600', marginBottom: 4 },
  aiStatusTitle: { fontSize: 14, fontWeight: '700', color: TEXT },
  aiStatusMeta: { fontSize: 12, color: MUTED, marginTop: 4, lineHeight: 17 },
  btnTxt: { color: '#fff', fontSize: 16, fontWeight: '700' },
  progressCard: { backgroundColor: '#fff', borderRadius: 12, padding: 18, marginBottom: 12, borderWidth: 1, borderColor: 'rgba(0,0,0,0.06)' },
  pct: { fontSize: 30, fontWeight: '800', color: ACCENT, textAlign: 'center' },
  track: { height: 8, backgroundColor: '#eee', borderRadius: 4, marginTop: 10, overflow: 'hidden' },
  fill: { height: 8, backgroundColor: ACCENT },
  counts: { fontSize: 13, color: TEXT, textAlign: 'center', marginTop: 10 },
  current: { fontSize: 11, color: MUTED, textAlign: 'center', marginTop: 4 },
  note: { marginTop: 16 },
  noteTxt: { fontSize: 12, color: MUTED, lineHeight: 20 },
  uploadResult: {
    backgroundColor: 'rgba(58,158,92,0.10)', borderRadius: 12, padding: 14, marginBottom: 10,
    borderWidth: 1, borderColor: 'rgba(58,158,92,0.30)',
  },
  uploadResultTxt: { fontSize: 15, fontWeight: '700', color: '#2e7d4f' },
  uploadResultErr: { fontSize: 12, color: MUTED, marginTop: 6 },
  uploadStat: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    backgroundColor: '#fff', borderRadius: 12, paddingVertical: 14, paddingHorizontal: 16,
    marginBottom: 14, borderWidth: 1, borderColor: 'rgba(0,0,0,0.08)',
  },
  uploadStatTxt: { fontSize: 15, fontWeight: '700', color: TEXT },
  uploadStatArrow: { fontSize: 14, color: ACCENT, fontWeight: '600' },
});
