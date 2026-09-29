import { useState, useEffect } from 'react';
import {
  View, Text, TextInput, TouchableOpacity, FlatList, Image, Modal,
  ActivityIndicator, StyleSheet, Dimensions,
} from 'react-native';
import { search, mediaUrl, originalUrl, BACKEND_URL } from '../api';
import ZoomableImage from '../components/ZoomableImage';
import OfflineNotice from '../components/OfflineNotice';
import { useGallery } from '../gallery';
import { ACCENT, BG, TEXT, MUTED, PADDING } from '../theme';

const COL = 2;
const SIZE = (Dimensions.get('window').width - PADDING * 2 - 10) / COL;

// 서버 도달 가능 여부(온라인) 확인 — 네이티브 모듈 없이 짧은 fetch
async function checkOnline() {
  try {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 4000);
    const res = await fetch(`${BACKEND_URL}/`, { method: 'GET', signal: ctrl.signal });
    clearTimeout(t);
    return res.status > 0;
  } catch { return false; }
}

export default function SearchScreen() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [parsed, setParsed] = useState(null);
  const [searched, setSearched] = useState(false);
  const [selected, setSelected] = useState(null);   // 상세보기(라이트박스)
  const [online, setOnline] = useState(null);        // null=확인중, true/false
  const { localUriFor } = useGallery();              // 폰 갤러리 원본 우선

  useEffect(() => { checkOnline().then(setOnline); }, []);

  const doSearch = async () => {
    const q = query.trim();
    if (!q) return;
    const net = await checkOnline();
    setOnline(net);
    if (!net) return;   // 오프라인 → 배너 안내, 검색 시도 안 함
    setLoading(true); setSearched(true);
    const data = await search(q);
    setResults(data?.results || []);
    setParsed(data?.parsed || null);
    setLoading(false);
  };

  return (
    <View style={styles.wrap}>
      <Text style={styles.title}>검색</Text>
      <Text style={styles.sub}>자연어로 원하는 순간을 찾아보세요</Text>

      {online === false ? (
        <OfflineNotice
          title="오프라인 상태예요"
          message="검색은 AI 서버(인터넷)가 필요해요. 다른 탭(프로필·인물·라이프)은 저장된 데이터로 계속 볼 수 있어요."
        />
      ) : (
       <>
      <View style={styles.searchBox}>
        <Text style={styles.searchIcon}>🔍</Text>
        <TextInput
          style={styles.input}
          value={query}
          onChangeText={setQuery}
          onSubmitEditing={doSearch}
          placeholder="예: 작년 여름 바다, 카페에서 친구랑..."
          placeholderTextColor={MUTED}
          returnKeyType="search"
        />
        {!!query && (
          <TouchableOpacity onPress={() => setQuery('')}>
            <Text style={{ color: MUTED, fontSize: 16 }}>✕</Text>
          </TouchableOpacity>
        )}
      </View>
      <TouchableOpacity style={styles.searchBtn} onPress={doSearch}>
        <Text style={styles.searchBtnTxt}>검색</Text>
      </TouchableOpacity>

      {/* LLM 이 해석한 검색 조건 */}
      {parsed && (
        <View style={styles.parsedBox}>
          <Text style={styles.parsedTxt}>
            해석: {[
              parsed.date_range && '날짜',
              (parsed.date_months?.length) && '월',
              parsed.location && '위치',
              (parsed.people?.length) && '인물',
              (parsed.activity_terms?.length) && '활동',
              (parsed.object_terms?.length) && '객체',
            ].filter(Boolean).join(' · ') || '(조건 없음)'}
          </Text>
        </View>
      )}

      {loading ? (
        <ActivityIndicator color={ACCENT} style={{ marginTop: 40 }} />
      ) : results.length > 0 ? (
        <FlatList
          data={results}
          keyExtractor={(it, i) => String(it.id ?? i)}
          numColumns={COL}
          columnWrapperStyle={{ gap: 10 }}
          contentContainerStyle={{ gap: 10, paddingVertical: 12 }}
          ListHeaderComponent={<Text style={styles.count}>{results.length}건</Text>}
          renderItem={({ item }) => {
            // 폰 갤러리에 있으면 폰 원본, 없으면 백엔드 썸네일로 폴백
            const uri = localUriFor(item.filename) || mediaUrl(item.thumbnail_path);
            return (
              <TouchableOpacity style={styles.card} activeOpacity={0.8} onPress={() => setSelected(item)}>
                {uri ? (
                  <Image source={{ uri }} style={styles.thumb} />
                ) : (
                  <View style={[styles.thumb, styles.noThumb]}><Text style={{ color: MUTED }}>이미지</Text></View>
                )}
                <Text style={styles.cardName} numberOfLines={1}>{item.filename || ''}</Text>
                {!!(item.match_reasons?.length) && (
                  <Text style={styles.cardReason}>{item.match_reasons.join(', ')}</Text>
                )}
              </TouchableOpacity>
            );
          }}
        />
      ) : searched ? (
        <View style={styles.empty}><Text style={styles.emptyTxt}>결과 없음</Text></View>
      ) : (
        <View style={styles.empty}><Text style={styles.emptyTxt}>검색어를 입력하세요</Text></View>
      )}
       </>
      )}

      {/* 상세보기 (라이트박스) */}
      <Modal visible={!!selected} transparent animationType="fade" onRequestClose={() => setSelected(null)}>
        <View style={styles.modalBg}>
          <TouchableOpacity style={styles.modalClose} onPress={() => setSelected(null)}>
            <Text style={{ color: '#fff', fontSize: 22 }}>✕</Text>
          </TouchableOpacity>
          {selected && (
            <>
              <ZoomableImage style={styles.modalImg}>
                <Image
                  source={{ uri: localUriFor(selected.filename) || originalUrl(selected.filename) }}
                  style={{ width: '100%', height: '100%' }}
                  resizeMode="contain"
                />
              </ZoomableImage>
              <View style={styles.modalInfo}>
                <Text style={styles.modalName}>{selected.filename}</Text>
                {!!selected.datetime && <Text style={styles.modalMeta}>📅 {String(selected.datetime).slice(0, 16).replace('T', ' ')}</Text>}
                {selected.lat != null && <Text style={styles.modalMeta}>📍 {selected.lat?.toFixed(5)}, {selected.lon?.toFixed(5)}</Text>}
                {!!(selected.match_reasons?.length) && <Text style={styles.modalMeta}>🏷️ {selected.match_reasons.join(', ')}</Text>}
              </View>
            </>
          )}
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { flex: 1, backgroundColor: BG, paddingHorizontal: PADDING },
  title: { fontSize: 24, fontWeight: '700', color: TEXT, marginTop: 8 },
  sub: { fontSize: 13, color: MUTED, marginTop: 4, marginBottom: 14 },
  offlineBanner: { backgroundColor: 'rgba(224,108,117,0.1)', borderWidth: 1, borderColor: 'rgba(224,108,117,0.3)', borderRadius: 10, padding: 12, marginBottom: 14 },
  offlineTitle: { fontSize: 14, fontWeight: '700', color: '#c0392b' },
  offlineSub: { fontSize: 12, color: '#7a4a46', marginTop: 4, lineHeight: 17 },
  searchBox: {
    flexDirection: 'row', alignItems: 'center', backgroundColor: '#fff',
    borderRadius: 12, paddingHorizontal: 12, height: 48,
    borderWidth: 1, borderColor: 'rgba(0,0,0,0.08)',
  },
  searchIcon: { fontSize: 16, marginRight: 8 },
  input: { flex: 1, fontSize: 15, color: TEXT },
  searchBtn: { backgroundColor: ACCENT, borderRadius: 12, height: 44, justifyContent: 'center', alignItems: 'center', marginTop: 10 },
  searchBtnTxt: { color: '#fff', fontSize: 15, fontWeight: '600' },
  parsedBox: { backgroundColor: 'rgba(200,90,48,0.08)', borderRadius: 8, padding: 10, marginTop: 12 },
  parsedTxt: { fontSize: 12, color: ACCENT },
  count: { fontSize: 13, color: MUTED, marginBottom: 4 },
  card: { width: SIZE },
  thumb: { width: SIZE, height: SIZE, borderRadius: 10, backgroundColor: '#eee' },
  noThumb: { justifyContent: 'center', alignItems: 'center' },
  cardName: { fontSize: 11, color: TEXT, marginTop: 4 },
  cardReason: { fontSize: 10, color: ACCENT },
  empty: { flex: 1, justifyContent: 'center', alignItems: 'center', marginTop: 40 },
  emptyTxt: { color: MUTED, fontSize: 14 },
  // 라이트박스 모달
  modalBg: { flex: 1, backgroundColor: 'rgba(0,0,0,0.92)', justifyContent: 'center' },
  modalClose: { position: 'absolute', top: 44, right: 20, zIndex: 2, padding: 8 },
  modalImg: { width: '100%', height: '70%' },
  modalInfo: { paddingHorizontal: PADDING, paddingTop: 16 },
  modalName: { color: '#fff', fontSize: 15, fontWeight: '600', marginBottom: 6 },
  modalMeta: { color: '#ccc', fontSize: 13, marginTop: 3 },
});
