// 얼굴 crop 이미지 — 서버 이미지라 오프라인이면 👤 플레이스홀더로 통일.
// (온라인일 때만 표시, 로드 실패 시에도 👤)
import { useState } from 'react';
import { View, Text, Image, StyleSheet } from 'react-native';
import { useOnline } from '../useOnline';

export default function FaceImage({ uri, style, emojiSize = 22 }) {
  const [err, setErr] = useState(false);
  const online = useOnline();
  if (!uri || err || online === false) {
    return (
      <View style={[style, styles.ph]}>
        <Text style={{ fontSize: emojiSize }}>👤</Text>
      </View>
    );
  }
  return <Image source={{ uri }} style={style} onError={() => setErr(true)} />;
}

const styles = StyleSheet.create({
  ph: { backgroundColor: '#e3e1dd', justifyContent: 'center', alignItems: 'center' },
});
