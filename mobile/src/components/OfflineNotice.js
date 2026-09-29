// 오프라인 안내 — 화면 정중앙, wifi-off 아이콘 + 메시지 (공용)
import { View, Text, StyleSheet } from 'react-native';
import { BG, TEXT, MUTED } from '../theme';

function WifiOff({ size = 64 }) {
  return (
    <View style={{ width: size, height: size, justifyContent: 'center', alignItems: 'center', marginBottom: 16 }}>
      <Text style={{ fontSize: size * 0.82 }}>📶</Text>
      <View style={{
        position: 'absolute', width: size * 1.15, height: 5, borderRadius: 3,
        backgroundColor: '#e06c75', transform: [{ rotate: '-45deg' }],
      }} />
    </View>
  );
}

export default function OfflineNotice({
  title = '오프라인 상태예요',
  message = '이 기능은 인터넷 연결이 필요해요.',
}) {
  return (
    <View style={styles.wrap}>
      <WifiOff />
      <Text style={styles.title}>{title}</Text>
      <Text style={styles.msg}>{message}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: BG, paddingHorizontal: 36 },
  title: { fontSize: 18, fontWeight: '700', color: TEXT, textAlign: 'center' },
  msg: { fontSize: 14, color: MUTED, textAlign: 'center', marginTop: 8, lineHeight: 21 },
});
