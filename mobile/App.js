import { useState, useEffect } from 'react';
import {
  View, Text, TouchableOpacity, SafeAreaView, StatusBar, StyleSheet, Platform,
} from 'react-native';
import * as NavigationBar from 'expo-navigation-bar';
import { ACCENT, BG, TEXT, MUTED, BORDER, PADDING } from './src/theme';
import { GalleryProvider } from './src/gallery';
import { getSavedUser, setApiUser, clearSession } from './src/api';
import AuthScreen from './src/screens/AuthScreen';
import SearchScreen from './src/screens/SearchScreen';
import ProfileScreen from './src/screens/ProfileScreen';
import LifestyleScreen from './src/screens/LifestyleScreen';
import MapScreen from './src/screens/MapScreen';
import GraphScreen from './src/screens/GraphScreen';
import PeopleScreen from './src/screens/PeopleScreen';
import UploadScreen from './src/screens/UploadScreen';
import TimePredictScreen from './src/screens/TimePredictScreen';

const TABS = [
  { key: 'search',    label: '검색',    icon: '🔍', screen: SearchScreen },
  { key: 'profile',   label: '프로필',  icon: '🧬', screen: ProfileScreen },
  { key: 'lifestyle', label: '라이프',  icon: '📷', screen: LifestyleScreen },
  { key: 'map',       label: '지도',    icon: '🗺️', screen: MapScreen },
  { key: 'graph',     label: '관계',    icon: '🕸️', screen: GraphScreen },
  { key: 'people',    label: '인물',    icon: '👥', screen: PeopleScreen },
  { key: 'upload',    label: '업로드',  icon: '📤', screen: UploadScreen },
  { key: 'timepredict', label: '예측', icon: '⏱️', screen: TimePredictScreen },
];

export default function App() {
  const [authedUser, setAuthedUser] = useState(undefined);  // undefined=확인중, null=미로그인, string=로그인됨
  const [tab, setTab] = useState('search');
  const [pendingPerson, setPendingPerson] = useState(null);  // 라이프→인물 탭으로 열 인물
  const ActiveScreen = TABS.find(t => t.key === tab).screen;

  // 저장된 로그인 세션 복원
  useEffect(() => {
    getSavedUser().then(uid => {
      if (uid) { setApiUser(uid); setAuthedUser(uid); }
      else setAuthedUser(null);
    }).catch(() => setAuthedUser(null));
  }, []);

  // 안드로이드 하단 시스템 메뉴바(소프트키) 숨기기 (몰입 모드)
  useEffect(() => {
    if (Platform.OS === 'android') {
      const hideNav = async () => {
        try {
          if (typeof NavigationBar.setHidden === 'function') {
            await NavigationBar.setHidden(true);
          } else if (typeof NavigationBar.setVisibilityAsync === 'function') {
            await NavigationBar.setVisibilityAsync('hidden');
          }
        } catch (e) {
          console.log('NavigationBar hide error:', e);
        }
      };
      setTimeout(hideNav, 500);
    }
  }, []);

  // 다른 탭(라이프스타일)에서 인물 프로필 열기 → 인물 탭으로 전환 + 해당 인물 상세
  const openPersonInPeople = (pid) => { setPendingPerson(pid); setTab('people'); };

  const handleLogout = async () => {
    await clearSession();
    setTab('search');
    setAuthedUser(null);
  };

  // 세션 확인 중 — 빈 화면
  if (authedUser === undefined) {
    return <SafeAreaView style={styles.safe}><StatusBar barStyle="dark-content" backgroundColor={BG} hidden={Platform.OS === 'android'} /></SafeAreaView>;
  }

  // 미로그인 — 로그인/회원가입 화면
  if (!authedUser) {
    return (
      <SafeAreaView style={styles.safe}>
        <StatusBar barStyle="dark-content" backgroundColor={BG} hidden={Platform.OS === 'android'} />
        <AuthScreen onAuthed={(uid) => { setTab('upload'); setAuthedUser(uid); }} />
      </SafeAreaView>
    );
  }

  return (
    <GalleryProvider>
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="dark-content" backgroundColor={BG} hidden={Platform.OS === 'android'} />

      {/* 헤더 (기존 로고 스타일 유지) */}
      <View style={styles.header}>
        <Text style={styles.logo}>Lu<Text style={{ color: ACCENT }}>mina</Text></Text>
      </View>

      {/* 활성 화면 */}
      <View style={styles.body}>
        <ActiveScreen
          onOpenPerson={openPersonInPeople}
          openPersonId={pendingPerson}
          onConsumePerson={() => setPendingPerson(null)}
          userId={authedUser}
          onLogout={handleLogout}
        />
      </View>

      {/* 하단 탭 바 */}
      <View style={styles.tabBar}>
        {TABS.map(t => {
          const active = t.key === tab;
          return (
            <TouchableOpacity key={t.key} style={styles.tabBtn} onPress={() => setTab(t.key)} activeOpacity={0.7}>
              <Text style={[styles.tabIcon, { opacity: active ? 1 : 0.5 }]}>{t.icon}</Text>
              <Text style={[styles.tabLabel, active && { color: ACCENT, fontWeight: '700' }]}>{t.label}</Text>
            </TouchableOpacity>
          );
        })}
      </View>
    </SafeAreaView>
    </GalleryProvider>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: BG },
  header: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingHorizontal: PADDING, paddingVertical: 14,
    borderBottomWidth: 1, borderBottomColor: BORDER,
  },
  logo: {
    fontFamily: Platform.OS === 'ios' ? 'Georgia' : 'serif',
    fontSize: 22, fontWeight: '700', letterSpacing: -0.5, color: TEXT,
  },
  body: { flex: 1 },
  tabBar: {
    flexDirection: 'row', borderTopWidth: 1, borderTopColor: BORDER,
    backgroundColor: '#fff', paddingBottom: Platform.OS === 'ios' ? 18 : 6, paddingTop: 6,
  },
  tabBtn: { flex: 1, alignItems: 'center', justifyContent: 'center', paddingVertical: 4 },
  tabIcon: { fontSize: 20 },
  tabLabel: { fontSize: 11, color: MUTED, marginTop: 2 },
});
