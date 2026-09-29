// ─────────────────────────────────────────────────────────
//  ConsentModal — 개인정보 수집·이용 동의 화면 (Google Play Prominent Disclosure)
//  사진 업로드(수집)가 시작되기 전에 반드시 1회 동의를 받는다.
//  동의 기록은 AsyncStorage 에 유저별로 저장 (lumina_consent_v1_<uid>).
// ─────────────────────────────────────────────────────────
import { useState } from 'react';
import {
  Modal, View, Text, ScrollView, TouchableOpacity, StyleSheet,
} from 'react-native';
import { ACCENT, BG, TEXT, MUTED, PADDING } from '../theme';

export const CONSENT_VERSION = 'v1';
export const consentKey = (uid) => `lumina_consent_${CONSENT_VERSION}_${uid}`;

const SECTIONS = [
  {
    title: '1. 수집하는 정보',
    body:
      '• 회원님이 선택한 기기 갤러리의 사진(축소본)\n' +
      '• 사진에 포함된 촬영 일시·위치(GPS) 정보\n' +
      '• 로그인 계정 ID, 푸시 알림 토큰',
  },
  {
    title: '2. 이용 목적',
    body:
      '업로드된 사진은 AI 분석 서버로 전송되어 다음 목적에만 사용됩니다.\n' +
      '• 인물 인식 및 인물별 사진 분류\n' +
      '• 인물 관계·라이프스타일·타임라인 분석\n' +
      '• 사진 촬영 시각 예측',
  },
  {
    title: '3. 얼굴 인식(생체정보) 처리 안내',
    body:
      '인물 분류를 위해 사진 속 얼굴의 특징값(임베딩)이 생성·이용됩니다. ' +
      '이 정보는 분석 목적 외에는 사용되지 않으며, 외부에 판매·제공되지 않습니다.',
  },
  {
    title: '4. 처리 위탁 (제3자 전송)',
    body:
      '사진 내용 설명 생성 및 분석을 위해 사진이 Google Cloud(Gemini AI)에 ' +
      '전송·처리될 수 있습니다. Google은 위탁 목적 범위에서만 데이터를 처리합니다.',
  },
  {
    title: '5. 보관 및 삭제',
    body:
      '• AI 분석이 완료되면 서버에 업로드된 원본 사진은 일괄 삭제됩니다.\n' +
      '• 분석 결과(인물 분류·분석 리포트)는 서비스 제공을 위해 보관됩니다.\n' +
      '• "업로드 관리" 메뉴에서 언제든지 업로드한 사진을 직접 삭제할 수 있습니다.',
  },
  {
    title: '6. 이용자 권리',
    body:
      '회원님은 언제든지 수집된 사진의 열람·삭제를 요청할 수 있으며, ' +
      '동의를 거부할 수 있습니다. 다만 동의하지 않을 경우 사진 분석 서비스를 ' +
      '이용할 수 없습니다.',
  },
];

export default function ConsentModal({ visible, onAgree, onDecline }) {
  const [checked, setChecked] = useState(false);

  return (
    <Modal visible={visible} animationType="slide" onRequestClose={onDecline}>
      <View style={styles.wrap}>
        <Text style={styles.title}>개인정보 수집·이용 동의</Text>
        <Text style={styles.sub}>
          Lumina 의 사진 분석 서비스를 이용하려면 아래 내용을 확인하고 동의해 주세요.
        </Text>

        <ScrollView style={styles.scroll} contentContainerStyle={{ paddingBottom: 16 }}>
          {SECTIONS.map((s) => (
            <View key={s.title} style={styles.section}>
              <Text style={styles.secTitle}>{s.title}</Text>
              <Text style={styles.secBody}>{s.body}</Text>
            </View>
          ))}
        </ScrollView>

        <TouchableOpacity
          style={styles.checkRow}
          onPress={() => setChecked(!checked)}
          activeOpacity={0.7}
        >
          <View style={[styles.checkbox, checked && styles.checkboxOn]}>
            {checked && <Text style={styles.checkMark}>✓</Text>}
          </View>
          <Text style={styles.checkLabel}>
            위 내용을 모두 확인했으며, 사진 수집·이용 및 얼굴 인식 처리에 동의합니다.
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.agreeBtn, !checked && styles.btnDisabled]}
          disabled={!checked}
          onPress={onAgree}
        >
          <Text style={styles.agreeTxt}>동의하고 시작하기</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.declineBtn} onPress={onDecline}>
          <Text style={styles.declineTxt}>동의하지 않음</Text>
        </TouchableOpacity>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  wrap: { flex: 1, backgroundColor: BG, paddingHorizontal: PADDING, paddingTop: 24, paddingBottom: 20 },
  title: { fontSize: 22, fontWeight: '800', color: TEXT },
  sub: { fontSize: 13, color: MUTED, marginTop: 6, marginBottom: 14, lineHeight: 19 },
  scroll: {
    flex: 1, backgroundColor: '#fff', borderRadius: 12, padding: 14,
    borderWidth: 1, borderColor: 'rgba(0,0,0,0.08)',
  },
  section: { marginBottom: 14 },
  secTitle: { fontSize: 14, fontWeight: '700', color: TEXT, marginBottom: 4 },
  secBody: { fontSize: 13, color: TEXT, lineHeight: 20 },
  checkRow: { flexDirection: 'row', alignItems: 'center', marginTop: 14 },
  checkbox: {
    width: 24, height: 24, borderRadius: 6, borderWidth: 2, borderColor: MUTED,
    justifyContent: 'center', alignItems: 'center', marginRight: 10, backgroundColor: '#fff',
  },
  checkboxOn: { backgroundColor: ACCENT, borderColor: ACCENT },
  checkMark: { color: '#fff', fontWeight: '800', fontSize: 15 },
  checkLabel: { flex: 1, fontSize: 13, color: TEXT, lineHeight: 19 },
  agreeBtn: {
    backgroundColor: ACCENT, borderRadius: 12, height: 50,
    justifyContent: 'center', alignItems: 'center', marginTop: 14,
  },
  btnDisabled: { opacity: 0.4 },
  agreeTxt: { color: '#fff', fontSize: 16, fontWeight: '700' },
  declineBtn: { height: 44, justifyContent: 'center', alignItems: 'center', marginTop: 4 },
  declineTxt: { color: MUTED, fontSize: 14 },
});
