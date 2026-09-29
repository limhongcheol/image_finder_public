// 로그인 / 회원가입 — 앱 진입 게이트
import { useState } from 'react';
import {
  View, Text, TextInput, TouchableOpacity, ActivityIndicator,
  StyleSheet, Alert, KeyboardAvoidingView, Platform,
} from 'react-native';
import { register, login, saveSession } from '../api';
import { ACCENT, BG, TEXT, MUTED } from '../theme';

export default function AuthScreen({ onAuthed }) {
  const [mode, setMode] = useState('login');     // 'login' | 'register'
  const [userId, setUserId] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);

  const submit = async () => {
    const id = userId.trim();
    if (!id || !password) { Alert.alert('입력 필요', '아이디와 비밀번호를 입력하세요.'); return; }
    setLoading(true);
    try {
      if (mode === 'register') {
        const r = await register(id, password);
        if (!r.ok) { Alert.alert('회원가입 실패', r.data.error || '다시 시도해주세요.'); return; }
        // 가입 성공 → 이어서 자동 로그인
      }
      const lr = await login(id, password);
      if (!lr.ok) { Alert.alert('로그인 실패', lr.data.error || '다시 시도해주세요.'); return; }
      await saveSession(lr.data.user_id, lr.data.token);
      onAuthed(lr.data.user_id);
    } finally {
      setLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView
      style={styles.wrap}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <Text style={styles.logo}>Lu<Text style={{ color: ACCENT }}>mina</Text></Text>
      <Text style={styles.sub}>{mode === 'login' ? '로그인' : '회원가입'}</Text>

      <TextInput
        style={styles.input}
        value={userId}
        onChangeText={setUserId}
        placeholder="아이디 (영문/숫자/_, 3~32자)"
        placeholderTextColor={MUTED}
        autoCapitalize="none"
        autoCorrect={false}
      />
      <TextInput
        style={styles.input}
        value={password}
        onChangeText={setPassword}
        placeholder="비밀번호 (4자 이상)"
        placeholderTextColor={MUTED}
        secureTextEntry
        autoCapitalize="none"
      />

      <TouchableOpacity
        style={[styles.btn, loading && { opacity: 0.5 }]}
        onPress={submit}
        disabled={loading}
        activeOpacity={0.8}
      >
        {loading
          ? <ActivityIndicator color="#fff" />
          : <Text style={styles.btnTxt}>{mode === 'login' ? '로그인' : '회원가입'}</Text>}
      </TouchableOpacity>

      <TouchableOpacity onPress={() => setMode(mode === 'login' ? 'register' : 'login')} style={{ marginTop: 18 }}>
        <Text style={styles.toggle}>
          {mode === 'login' ? '계정이 없으신가요?  회원가입' : '이미 계정이 있으신가요?  로그인'}
        </Text>
      </TouchableOpacity>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  wrap: { flex: 1, backgroundColor: BG, justifyContent: 'center', paddingHorizontal: 28 },
  logo: { fontSize: 38, fontWeight: '800', color: TEXT, textAlign: 'center', letterSpacing: -1 },
  sub: { fontSize: 15, color: MUTED, textAlign: 'center', marginTop: 6, marginBottom: 32 },
  input: {
    height: 52, backgroundColor: '#fff', borderRadius: 12, paddingHorizontal: 16,
    fontSize: 16, color: TEXT, borderWidth: 1, borderColor: 'rgba(0,0,0,0.1)', marginBottom: 12,
  },
  btn: {
    backgroundColor: ACCENT, borderRadius: 12, height: 52,
    justifyContent: 'center', alignItems: 'center', marginTop: 6,
  },
  btnTxt: { color: '#fff', fontSize: 16, fontWeight: '700' },
  toggle: { color: ACCENT, fontSize: 14, textAlign: 'center', fontWeight: '600' },
  note: { color: MUTED, fontSize: 12, textAlign: 'center', marginTop: 28, lineHeight: 18 },
});
