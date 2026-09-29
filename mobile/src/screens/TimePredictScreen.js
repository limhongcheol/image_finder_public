// 촬영시각 예측 — 사진 1장 선택 → 서버(GPU 모델) 예측 → 결과 표시
import { useState } from 'react';
import {
  View, Text, TouchableOpacity, Image, ScrollView, ActivityIndicator,
  StyleSheet, Alert,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { ACCENT, BLUE, BG, TEXT, MUTED, BORDER, PADDING } from '../theme';
import { predictTime } from '../api';
import { useOnline } from '../useOnline';
import OfflineNotice from '../components/OfflineNotice';

export default function TimePredictScreen() {
  const online = useOnline();
  const [image, setImage] = useState(null);    // { uri, name }
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const pick = async (fromCamera) => {
    try {
      const perm = fromCamera
        ? await ImagePicker.requestCameraPermissionsAsync()
        : await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (!perm.granted) {
        Alert.alert('권한 필요', fromCamera ? '카메라 권한을 허용해주세요.' : '사진 접근 권한을 허용해주세요.');
        return;
      }
      const opts = { mediaTypes: ['images'], quality: 0.9 };
      const res = fromCamera
        ? await ImagePicker.launchCameraAsync(opts)
        : await ImagePicker.launchImageLibraryAsync(opts);
      if (res.canceled || !res.assets || !res.assets.length) return;
      const a = res.assets[0];
      setImage({ uri: a.uri, name: a.fileName || `photo_${Date.now()}.jpg` });
      setResult(null);
    } catch (e) {
      Alert.alert('오류', e.message);
    }
  };

  const run = async () => {
    if (!image || loading) return;
    setLoading(true);
    setResult(null);
    try {
      const r = await predictTime(image.uri, image.name);
      setResult(r);
    } catch (e) {
      Alert.alert('예측 실패', e.message || '잠시 후 다시 시도해주세요.');
    } finally {
      setLoading(false);
    }
  };

  if (online === false) {
    return <OfflineNotice title="오프라인 상태예요" message="촬영시각 예측은 인터넷 연결이 필요해요." />;
  }

  return (
    <ScrollView style={styles.wrap} contentContainerStyle={{ paddingBottom: 40 }}>
      <Text style={styles.title}>⏱️ 촬영시각 예측</Text>
      <Text style={styles.subtitle}>
        사진 한 장을 올리면 AI가 촬영 시각(월·일·시·분)을 예측합니다.
      </Text>

      {/* 선택 버튼 */}
      <View style={styles.pickRow}>
        <TouchableOpacity style={styles.pickBtn} onPress={() => pick(false)} activeOpacity={0.8}>
          <Text style={styles.pickIcon}>🖼️</Text>
          <Text style={styles.pickTxt}>갤러리</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.pickBtn} onPress={() => pick(true)} activeOpacity={0.8}>
          <Text style={styles.pickIcon}>📷</Text>
          <Text style={styles.pickTxt}>카메라</Text>
        </TouchableOpacity>
      </View>

      {/* 미리보기 */}
      {image && (
        <Image source={{ uri: image.uri }} style={styles.preview} resizeMode="cover" />
      )}

      {/* 예측 버튼 */}
      {image && (
        <TouchableOpacity
          style={[styles.runBtn, loading && { opacity: 0.5 }]}
          onPress={run}
          disabled={loading}
          activeOpacity={0.8}
        >
          <Text style={styles.runTxt}>{loading ? '예측 중…' : '예측하기'}</Text>
        </TouchableOpacity>
      )}

      {loading && (
        <View style={styles.loadingBox}>
          <ActivityIndicator color={ACCENT} />
          <Text style={styles.loadingTxt}>모델이 사진을 분석하고 있어요 (수~수십 초 소요)</Text>
        </View>
      )}

      {/* 결과 카드 */}
      {result && (
        <View style={styles.resultCard}>
          <Text style={styles.resultLabel}>예측 촬영 시각</Text>
          <Text style={styles.resultBig}>{result.predicted_time || '-'}</Text>
          <View style={styles.row}>
            <Text style={styles.rowKey}>전체 예측</Text>
            <Text style={styles.rowVal}>{result.predicted_time_full || '-'}</Text>
          </View>
          <View style={styles.row}>
            <Text style={styles.rowKey}>EXIF 실제 시각</Text>
            <Text style={styles.rowVal}>{result.actual_time || '(없음)'}</Text>
          </View>
          {!!result.caption && (
            <View style={styles.captionBox}>
              <Text style={styles.captionLabel}>AI 캡션</Text>
              <Text style={styles.captionTxt}>{result.caption}</Text>
            </View>
          )}
          <Text style={styles.note}>※ 연도는 예측 대상이 아니며 월·일·시·분만 추정합니다.</Text>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  wrap: { flex: 1, backgroundColor: BG, paddingHorizontal: PADDING },
  title: { fontSize: 24, fontWeight: '700', color: TEXT, marginTop: 12 },
  subtitle: { fontSize: 14, color: MUTED, marginTop: 6, lineHeight: 20 },

  pickRow: { flexDirection: 'row', gap: 12, marginTop: 20 },
  pickBtn: {
    flex: 1, alignItems: 'center', paddingVertical: 22, borderRadius: 14,
    borderWidth: 1, borderColor: BORDER, backgroundColor: '#fff',
  },
  pickIcon: { fontSize: 26 },
  pickTxt: { fontSize: 14, color: TEXT, fontWeight: '600', marginTop: 8 },

  preview: { width: '100%', height: 280, borderRadius: 14, marginTop: 20, backgroundColor: '#eceae6' },

  runBtn: {
    marginTop: 18, paddingVertical: 15, borderRadius: 12, backgroundColor: ACCENT,
    alignItems: 'center',
  },
  runTxt: { color: '#fff', fontSize: 16, fontWeight: '700' },

  loadingBox: { alignItems: 'center', marginTop: 20 },
  loadingTxt: { color: MUTED, fontSize: 13, marginTop: 10, textAlign: 'center' },

  resultCard: {
    marginTop: 22, padding: 20, borderRadius: 16,
    borderWidth: 1, borderColor: 'rgba(0,0,0,0.08)', backgroundColor: '#fff',
  },
  resultLabel: { fontSize: 13, color: MUTED },
  resultBig: { fontSize: 34, fontWeight: '800', color: ACCENT, marginTop: 4, letterSpacing: 1 },
  row: {
    flexDirection: 'row', justifyContent: 'space-between', marginTop: 12,
    borderTopWidth: 1, borderTopColor: 'rgba(0,0,0,0.05)', paddingTop: 10,
  },
  rowKey: { fontSize: 14, color: MUTED },
  rowVal: { fontSize: 14, color: TEXT, fontWeight: '600' },
  captionBox: {
    marginTop: 14, padding: 12, borderRadius: 10, backgroundColor: 'rgba(58,124,198,0.08)',
  },
  captionLabel: { fontSize: 12, color: BLUE, fontWeight: '700' },
  captionTxt: { fontSize: 14, color: TEXT, marginTop: 4, lineHeight: 20 },
  note: { fontSize: 12, color: MUTED, marginTop: 14 },
});
