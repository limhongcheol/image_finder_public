import { useRef } from 'react';
import { Animated, PanResponder, View } from 'react-native';

// 두 손가락 핀치 줌 + 확대 시 한 손가락 팬. 네이티브 모듈 없이 PanResponder 로 구현.
//   <ZoomableImage style={...}><Image .../></ZoomableImage>
export default function ZoomableImage({ children, style, maxScale = 4 }) {
  const scale = useRef(new Animated.Value(1)).current;
  const tx = useRef(new Animated.Value(0)).current;
  const ty = useRef(new Animated.Value(0)).current;
  const lastScale = useRef(1);
  const lastTx = useRef(0);
  const lastTy = useRef(0);
  const startDist = useRef(null);

  const dist = (t) => Math.hypot(t[0].pageX - t[1].pageX, t[0].pageY - t[1].pageY);

  const reset = () => {
    lastScale.current = 1; lastTx.current = 0; lastTy.current = 0;
    Animated.parallel([
      Animated.spring(scale, { toValue: 1, useNativeDriver: true }),
      Animated.spring(tx, { toValue: 0, useNativeDriver: true }),
      Animated.spring(ty, { toValue: 0, useNativeDriver: true }),
    ]).start();
  };

  const responder = useRef(
    PanResponder.create({
      // 단순 탭은 통과(닫기 버튼 등), 2손가락 또는 확대상태 이동만 캡처
      onStartShouldSetPanResponder: () => false,
      onMoveShouldSetPanResponder: (e) =>
        e.nativeEvent.touches.length === 2 || lastScale.current > 1.01,
      onPanResponderMove: (e, g) => {
        const t = e.nativeEvent.touches;
        if (t.length === 2) {
          const d = dist(t);
          if (startDist.current == null) startDist.current = d;
          let next = lastScale.current * (d / startDist.current);
          next = Math.max(1, Math.min(next, maxScale));
          scale.setValue(next);
        } else if (t.length === 1 && lastScale.current > 1.01) {
          tx.setValue(lastTx.current + g.dx);
          ty.setValue(lastTy.current + g.dy);
        }
      },
      onPanResponderRelease: () => {
        startDist.current = null;
        scale.stopAnimation((v) => {
          lastScale.current = v;
          if (v <= 1.01) reset();
        });
        tx.stopAnimation((v) => (lastTx.current = v));
        ty.stopAnimation((v) => (lastTy.current = v));
      },
    })
  ).current;

  return (
    <View style={style} {...responder.panHandlers}>
      <Animated.View
        style={{ flex: 1, transform: [{ scale }, { translateX: tx }, { translateY: ty }] }}
      >
        {children}
      </Animated.View>
    </View>
  );
}
