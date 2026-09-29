// 업로드 사진 관리 — 업로드됨/미업로드 썸네일 그리드 + 서버 삭제(폰 원본 유지)
// 서버는 파일명 목록만 주고, 썸네일·asset_id 는 폰 갤러리(useGallery)에서 해결 → 대량에서도 빠름.
import { useState, useEffect, useMemo } from 'react';
import {
  View, Text, TouchableOpacity, Image, FlatList, Modal, ActivityIndicator,
  StyleSheet, Alert, Dimensions,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { getUploads, deleteUpload, deleteAllUploads } from '../api';
import { useGallery } from '../gallery';
import { upKey } from '../uploadSync';
import { ACCENT, BG, TEXT, MUTED, BORDER } from '../theme';

const COLS = 3;
const GAP = 4;
const SIZE = Math.floor((Dimensions.get('window').width - GAP * (COLS + 1)) / COLS);

export default function UploadGalleryModal({ visible, onClose, userId, onChanged }) {
  const { allPhotos, localUriFor, assetIdFor } = useGallery();
  const [tab, setTab] = useState('uploaded');     // 'uploaded' | 'pending'
  const [uploads, setUploads] = useState(null);   // 서버 파일명 배열 (["a.jpg", ...])
  const [loading, setLoading] = useState(false);

  const load = async () => {
    setLoading(true);
    const r = await getUploads(userId);
    setUploads(Array.isArray(r?.files) ? r.files : []);
    setLoading(false);
  };
  useEffect(() => { if (visible) { setTab('uploaded'); load(); } }, [visible]);

  // 업로드된 파일명(소문자) 집합 → 미업로드 판별용
  const uploadedSet = useMemo(
    () => new Set((uploads || []).map(f => String(f).toLowerCase())),
    [uploads],
  );
  const pending = useMemo(
    () => allPhotos().filter(p => !uploadedSet.has(p.filename)),
    [uploadedSet, allPhotos],
  );

  // 삭제된 파일의 asset_id 를 upKey 에서 제거 (다시 업로드 대상이 되게)
  const removeFromUpKey = async (filename) => {
    const aid = assetIdFor(filename);
    if (!aid) return;
    try {
      const raw = await AsyncStorage.getItem(upKey(userId));
      if (raw) {
        const set = new Set(JSON.parse(raw));
        if (set.delete(aid)) await AsyncStorage.setItem(upKey(userId), JSON.stringify([...set]));
      }
    } catch {}
  };

  const confirmDelete = (filename) => {
    Alert.alert('사진 삭제', '이 사진을 서버에서 삭제할까요?\n(폰 원본은 그대로 유지됩니다)', [
      { text: '취소', style: 'cancel' },
      { text: '삭제', style: 'destructive', onPress: () => doDelete(filename) },
    ]);
  };
  const doDelete = async (filename) => {
    const r = await deleteUpload(userId, filename);
    if (!r.ok) { Alert.alert('삭제 실패', r.data?.error || '다시 시도해주세요.'); return; }
    setUploads(prev => (prev || []).filter(f => f !== filename));
    await removeFromUpKey(filename);
    onChanged && onChanged();
  };

  const confirmDeleteAll = () => {
    const n = (uploads || []).length;
    if (!n) return;
    Alert.alert('전체 삭제', `서버에 올라간 사진 ${n}장을 모두 삭제할까요?\n(폰 원본은 그대로 유지됩니다)`, [
      { text: '취소', style: 'cancel' },
      { text: '전체 삭제', style: 'destructive', onPress: doDeleteAll },
    ]);
  };
  const doDeleteAll = async () => {
    const r = await deleteAllUploads(userId);
    if (!r.ok) { Alert.alert('삭제 실패', r.data?.error || '다시 시도해주세요.'); return; }
    setUploads([]);
    try { await AsyncStorage.removeItem(upKey(userId)); } catch {}   // 로컬 추적 초기화 → 전부 재업로드 대상
    onChanged && onChanged();
    Alert.alert('완료', `${r.data?.deleted_count ?? ''}장이 삭제되었습니다.`);
  };

  const renderUploaded = ({ item }) => {   // item = 파일명 문자열
    const uri = localUriFor(item);
    return (
      <TouchableOpacity style={styles.cell} onPress={() => confirmDelete(item)} activeOpacity={0.8}>
        {uri
          ? <Image source={{ uri }} style={styles.img} />
          : <View style={[styles.img, styles.ph]}><Text style={styles.phTxt}>🖼️</Text></View>}
        <View style={styles.delBadge}><Text style={styles.delTxt}>✕</Text></View>
      </TouchableOpacity>
    );
  };
  const renderPending = ({ item }) => (
    <View style={styles.cell}><Image source={{ uri: item.uri }} style={styles.img} /></View>
  );

  return (
    <Modal visible={visible} animationType="slide" onRequestClose={onClose}>
      <View style={styles.wrap}>
        <View style={styles.header}>
          <Text style={styles.title}>업로드 사진 관리</Text>
          <View style={styles.headerRight}>
            {tab === 'uploaded' && uploads && uploads.length > 0 && (
              <TouchableOpacity onPress={confirmDeleteAll} hitSlop={{ top: 10, bottom: 10, left: 8, right: 8 }}>
                <Text style={styles.delAll}>전체삭제</Text>
              </TouchableOpacity>
            )}
            <TouchableOpacity onPress={onClose} hitSlop={{ top: 10, bottom: 10, left: 8, right: 8 }}>
              <Text style={styles.close}>닫기</Text>
            </TouchableOpacity>
          </View>
        </View>

        <View style={styles.tabs}>
          <TouchableOpacity style={[styles.tab, tab === 'uploaded' && styles.tabA]} onPress={() => setTab('uploaded')}>
            <Text style={[styles.tabTxt, tab === 'uploaded' && styles.tabTxtA]}>
              업로드됨 {uploads ? `(${uploads.length})` : ''}
            </Text>
          </TouchableOpacity>
          <TouchableOpacity style={[styles.tab, tab === 'pending' && styles.tabA]} onPress={() => setTab('pending')}>
            <Text style={[styles.tabTxt, tab === 'pending' && styles.tabTxtA]}>미업로드 ({pending.length})</Text>
          </TouchableOpacity>
        </View>

        {loading ? (
          <ActivityIndicator color={ACCENT} style={{ marginTop: 40 }} />
        ) : tab === 'uploaded' ? (
          <FlatList
            data={uploads || []} keyExtractor={(i) => i} numColumns={COLS}
            renderItem={renderUploaded} contentContainerStyle={styles.grid}
            initialNumToRender={30} windowSize={5}
            ListEmptyComponent={<Text style={styles.empty}>업로드된 사진이 없어요</Text>}
          />
        ) : (
          <FlatList
            data={pending} keyExtractor={(i) => i.filename} numColumns={COLS}
            renderItem={renderPending} contentContainerStyle={styles.grid}
            initialNumToRender={30} windowSize={5}
            ListEmptyComponent={<Text style={styles.empty}>미업로드 사진이 없어요</Text>}
          />
        )}

        {tab === 'uploaded' && uploads && uploads.length > 0 && (
          <Text style={styles.hint}>사진을 탭하면 서버에서 삭제됩니다 (폰 원본은 유지)</Text>
        )}
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  wrap: { flex: 1, backgroundColor: BG },
  header: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingHorizontal: 16, paddingTop: 50, paddingBottom: 12, borderBottomWidth: 1, borderBottomColor: BORDER,
  },
  title: { fontSize: 18, fontWeight: '700', color: TEXT },
  headerRight: { flexDirection: 'row', alignItems: 'center', gap: 16 },
  delAll: { fontSize: 15, color: '#d0402a', fontWeight: '700' },
  close: { fontSize: 15, color: ACCENT, fontWeight: '600' },
  tabs: { flexDirection: 'row', paddingHorizontal: 12, paddingVertical: 10, gap: 8 },
  tab: { flex: 1, paddingVertical: 10, borderRadius: 10, alignItems: 'center', backgroundColor: '#fff', borderWidth: 1, borderColor: BORDER },
  tabA: { backgroundColor: ACCENT, borderColor: ACCENT },
  tabTxt: { fontSize: 14, color: TEXT, fontWeight: '600' },
  tabTxtA: { color: '#fff' },
  grid: { padding: GAP },
  cell: { width: SIZE, height: SIZE, margin: GAP / 2 },
  img: { width: '100%', height: '100%', borderRadius: 6, backgroundColor: '#eceae6' },
  ph: { justifyContent: 'center', alignItems: 'center' },
  phTxt: { fontSize: 26, opacity: 0.4 },
  delBadge: {
    position: 'absolute', top: 4, right: 4, width: 20, height: 20, borderRadius: 10,
    backgroundColor: 'rgba(200,60,40,0.9)', justifyContent: 'center', alignItems: 'center',
  },
  delTxt: { color: '#fff', fontSize: 12, fontWeight: '700' },
  empty: { textAlign: 'center', color: MUTED, marginTop: 40, fontSize: 14 },
  hint: { textAlign: 'center', color: MUTED, fontSize: 12, paddingVertical: 10 },
});
