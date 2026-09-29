import { useEffect, useMemo, useRef, useState } from 'react';
import {
  View, Text, Image, ScrollView, FlatList, TouchableOpacity, ActivityIndicator, StyleSheet, Modal,
} from 'react-native';
import { getLifestyleEvents, getPhotoPeople, getPerson, personImageUrl, mediaUrl } from '../api';
import ZoomableImage from '../components/ZoomableImage';
import FaceImage from '../components/FaceImage';
import { useGallery } from '../gallery';
import { ACCENT, BLUE, BG, TEXT, MUTED, PADDING } from '../theme';

// 라이프스타일: 시간순 사진 브라우저 (연도/기간 통계 + 밀집도 + 등장인물 + 하단 시간바)
export default function LifestyleScreen({ onOpenPerson }) {
  const { localUriFor } = useGallery();   // 폰 갤러리 원본 우선(오프라인)
  const [loading, setLoading] = useState(true);
  const [events, setEvents] = useState([]);
  const [year, setYear] = useState(null);   // null = 전체
  const [idx, setIdx] = useState(0);
  const [photoPeople, setPhotoPeople] = useState([]);   // 현재 사진 등장 인물
  const [detail, setDetail] = useState(null);           // 인물 상세 (모달)
  const [detailLoading, setDetailLoading] = useState(false);
  const stripRef = useRef(null);

  useEffect(() => {
    (async () => {
      const d = await getLifestyleEvents();
      const evs = (d?.events || [])
        .filter(e => e.datetime)
        .sort((a, b) => (a.timestamp_ms || 0) - (b.timestamp_ms || 0));
      setEvents(evs);
      setLoading(false);
    })();
  }, []);

  const years = useMemo(() => {
    const s = new Set(events.map(e => (e.datetime || '').slice(0, 4)).filter(Boolean));
    return [...s].sort();
  }, [events]);

  const filtered = useMemo(
    () => (year ? events.filter(e => (e.datetime || '').startsWith(year)) : events),
    [events, year],
  );

  // 밀집도: 연도 선택 시 월별, 전체일 때 연도별
  const density = useMemo(() => {
    const m = {};
    filtered.forEach(e => {
      const key = year ? (e.datetime || '').slice(5, 7) : (e.datetime || '').slice(0, 4);
      if (key) m[key] = (m[key] || 0) + 1;
    });
    return Object.entries(m).sort((a, b) => a[0].localeCompare(b[0]));
  }, [filtered, year]);

  useEffect(() => { setIdx(0); }, [year]);

  // 현재 선택 사진의 등장 인물(이름·얼굴·관계) 조회
  const curFilename = filtered[Math.min(idx, Math.max(0, filtered.length - 1))]?.filename;
  useEffect(() => {
    if (!curFilename) { setPhotoPeople([]); return; }
    let alive = true;
    getPhotoPeople(curFilename).then(d => { if (alive) setPhotoPeople(d?.people || []); });
    return () => { alive = false; };
  }, [curFilename]);

  if (loading) return <View style={styles.center}><ActivityIndicator color={ACCENT} /></View>;
  if (!filtered.length) return <View style={styles.center}><Text style={{ color: MUTED }}>사진 데이터 없음</Text></View>;

  const cur = filtered[Math.min(idx, filtered.length - 1)] || {};
  const a = cur.analysis || {};
  const ppl = a.people || {};
  const maxD = Math.max(1, ...density.map(d => d[1]));

  // 현재 사진과 같은 날(YYYY-MM-DD) 사진들만
  const dayKey = (cur.datetime || '').slice(0, 10);
  const sameDay = filtered.filter(e => (e.datetime || '').slice(0, 10) === dayKey);

  const selectIdx = (i) => {
    if (i < 0 || i >= filtered.length) return;
    setIdx(i);
  };

  // 밀집도 막대 탭 → 그 달/연도의 첫 사진으로 이동
  const jumpToKey = (key) => {
    const i = filtered.findIndex(e => {
      const dt = e.datetime || '';
      return year ? dt.slice(5, 7) === key : dt.slice(0, 4) === key;
    });
    if (i >= 0) selectIdx(i);
  };

  // 인물 카드 탭 → 상세 프로필 모달
  const openPersonDetail = async (pid) => {
    const pObj = photoPeople.find(x => x.person_id === pid) || {};
    setDetailLoading(true);
    setDetail({ person_id: pid, name: pObj.name, relationship_type: pObj.relationship_type });
    const d = await getPerson(pid);
    setDetail({ ...(d || { person_id: pid, images: [] }), name: pObj.name, relationship_type: pObj.relationship_type });
    setDetailLoading(false);
  };

  return (
    <View style={styles.wrap}>
      {/* 연도 필터 */}
      <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.yearBar} contentContainerStyle={{ gap: 6, paddingHorizontal: PADDING, alignItems: 'center' }}>
        <YChip label="전체" active={!year} onPress={() => setYear(null)} />
        {years.map(y => <YChip key={y} label={y} active={year === y} onPress={() => setYear(y)} />)}
      </ScrollView>

      {/* 밀집도 (탭하면 그 달/연도 사진으로) */}
      <View style={styles.densityRow}>
        <Text style={styles.densityLabel}>밀집도</Text>
        <View style={styles.densityBars}>
          {density.map(([k, v]) => {
            const isCur = year
              ? (cur.datetime || '').slice(5, 7) === k
              : (cur.datetime || '').slice(0, 4) === k;
            return (
              <TouchableOpacity key={k} style={styles.densityCol} onPress={() => jumpToKey(k)} activeOpacity={0.6}>
                <Text style={styles.densityCount}>{v}</Text>
                <View style={[styles.densityFill, { height: 4 + (v / maxD) * 24 }, isCur && styles.densityFillCur]} />
                <Text style={[styles.densityTick, isCur && { color: ACCENT, fontWeight: '700' }]}>{year ? +k : k.slice(2)}</Text>
              </TouchableOpacity>
            );
          })}
        </View>
      </View>

      {/* 메인 사진 (두손 줌) + 이전/다음 화살표 */}
      <View style={styles.main}>
        {(localUriFor(cur.filename) || cur.thumbnail_path)
          ? <ZoomableImage style={styles.mainImg}>
              <Image source={{ uri: localUriFor(cur.filename) || mediaUrl(cur.thumbnail_path) }} style={{ width: '100%', height: '100%' }} resizeMode="contain" />
            </ZoomableImage>
          : <View style={styles.mainImg} />}

        <TouchableOpacity style={[styles.arrow, styles.arrowLeft]} disabled={idx <= 0} onPress={() => selectIdx(idx - 1)} activeOpacity={0.6}>
          <Text style={[styles.arrowTxt, idx <= 0 && { opacity: 0.25 }]}>‹</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.arrow, styles.arrowRight]} disabled={idx >= filtered.length - 1} onPress={() => selectIdx(idx + 1)} activeOpacity={0.6}>
          <Text style={[styles.arrowTxt, idx >= filtered.length - 1 && { opacity: 0.25 }]}>›</Text>
        </TouchableOpacity>

        <View style={styles.overlay}>
          <Text style={styles.dt}>{(cur.datetime || '').replace('T', ' ').slice(0, 16)}</Text>
          {(cur.lat != null && cur.lon != null)
            ? <Text style={styles.loc}>📍 {Number(cur.lat).toFixed(3)}, {Number(cur.lon).toFixed(3)}</Text>
            : null}
        </View>
      </View>

      {/* 등장 인물 + 분석 */}
      <ScrollView style={styles.info} contentContainerStyle={{ paddingVertical: 8, paddingHorizontal: PADDING }}>
        <Text style={styles.infoTitle}>
          👥 등장 인물 {photoPeople.length ? `(${photoPeople.length})` : ''}
        </Text>
        {photoPeople.length > 0 ? (
          <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={{ gap: 10, paddingVertical: 6 }}>
            {photoPeople.map(p => (
              <TouchableOpacity key={p.person_id} style={styles.personCard} activeOpacity={0.7} onPress={() => (onOpenPerson ? onOpenPerson(p.person_id) : openPersonDetail(p.person_id))}>
                <FaceImage uri={p.thumbnail ? mediaUrl(p.thumbnail) : null} style={styles.personFace} emojiSize={20} />
                <Text style={styles.personName} numberOfLines={1}>{p.name}</Text>
                {!!p.relationship_type && <Text style={styles.personRel} numberOfLines={1}>{p.relationship_type}</Text>}
              </TouchableOpacity>
            ))}
          </ScrollView>
        ) : (
          <Text style={styles.peopleLine}>
            {ppl.count_estimate != null ? `분석상 ${ppl.count_estimate}명${ppl.group_type ? ' · ' + ppl.group_type : ''}` : '등장 인물 정보 없음'}
          </Text>
        )}
        <View style={styles.tags}>
          {a.activity?.label ? <Tag c="#22c55e" t={`활동 · ${a.activity.label}`} /> : null}
          {a.mood?.atmosphere ? <Tag c="#ec4899" t={a.mood.atmosphere} /> : null}
          {a.food?.present ? <Tag c="#fb8500" t={`음식${a.food.cuisine && a.food.cuisine !== 'unknown' ? ' · ' + a.food.cuisine : ''}`} /> : null}
          {a.landscape?.present ? <Tag c="#06b6d4" t={`풍경${a.landscape.type ? ' · ' + a.landscape.type : ''}`} /> : null}
          {a.activity?.indoor_outdoor ? <Tag c={BLUE} t={a.activity.indoor_outdoor} /> : null}
        </View>
        {a.landscape?.description ? <Text style={styles.desc}>{a.landscape.description}</Text> : null}
      </ScrollView>

      {/* 하단: 같은 날 사진들 */}
      <View style={styles.dayHeader}>
        <Text style={styles.dayHeaderTxt}>📅 {dayKey || '날짜 없음'} · 같은 날 {sameDay.length}장</Text>
        <Text style={styles.counter}>{idx + 1} / {filtered.length}</Text>
      </View>
      <FlatList
        data={sameDay}
        horizontal
        showsHorizontalScrollIndicator={false}
        keyExtractor={(it, i) => String(it.id ?? i)}
        style={styles.strip}
        contentContainerStyle={{ paddingHorizontal: 8, gap: 4, alignItems: 'center' }}
        renderItem={({ item }) => {
          const fi = filtered.indexOf(item);
          return (
            <TouchableOpacity onPress={() => selectIdx(fi)} activeOpacity={0.7}>
              <Image
                source={{ uri: localUriFor(item.filename) || mediaUrl(item.thumbnail_path) }}
                style={[styles.thumb, fi === idx && styles.thumbActive]}
              />
            </TouchableOpacity>
          );
        }}
      />

      {/* 인물 상세 프로필 모달 */}
      <Modal visible={!!detail} animationType="slide" transparent onRequestClose={() => setDetail(null)}>
        <View style={styles.modalOverlay}>
          <View style={styles.modalCard}>
            <View style={styles.modalHead}>
              <View style={{ flex: 1 }}>
                <Text style={styles.modalTitle}>{detail?.name || detail?.person_id}</Text>
                {!!detail?.relationship_type && <Text style={styles.modalRel}>{detail.relationship_type}</Text>}
              </View>
              <TouchableOpacity onPress={() => setDetail(null)}><Text style={styles.modalClose}>✕</Text></TouchableOpacity>
            </View>
            {detailLoading ? (
              <ActivityIndicator color={ACCENT} style={{ padding: 30 }} />
            ) : (
              <ScrollView contentContainerStyle={{ paddingBottom: 20 }}>
                <Text style={styles.modalMeta}>사진 {detail?.photo_count ?? 0}장</Text>
                <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={{ gap: 8, paddingVertical: 10 }}>
                  {(detail?.images || []).slice(0, 12).map((fn, i) => (
                    <FaceImage key={i} uri={personImageUrl(detail.person_id, fn)} style={styles.modalFace} />
                  ))}
                </ScrollView>
                {!!(detail?.level3 && Object.keys(detail.level3).length) && (
                  <View style={styles.modalSection}>
                    <Text style={styles.modalSectionTitle}>관계 (level3)</Text>
                    <Text style={styles.modalJson}>{JSON.stringify(detail.level3, null, 2).slice(0, 700)}</Text>
                  </View>
                )}
                {!!(detail?.level2 && Object.keys(detail.level2).length) && (
                  <View style={styles.modalSection}>
                    <Text style={styles.modalSectionTitle}>패턴 (level2)</Text>
                    <Text style={styles.modalJson}>{JSON.stringify(detail.level2, null, 2).slice(0, 700)}</Text>
                  </View>
                )}
              </ScrollView>
            )}
          </View>
        </View>
      </Modal>
    </View>
  );
}

const YChip = ({ label, active, onPress }) => (
  <TouchableOpacity onPress={onPress} style={[styles.ychip, active && styles.ychipActive]} activeOpacity={0.7}>
    <Text style={[styles.ychipTxt, active && styles.ychipTxtActive]}>{label}</Text>
  </TouchableOpacity>
);

const Tag = ({ c, t }) => (
  <Text style={[styles.tag, { color: c, borderColor: c }]}>{t}</Text>
);

const styles = StyleSheet.create({
  wrap: { flex: 1, backgroundColor: BG },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: BG, padding: 40 },

  yearBar: { maxHeight: 48, flexGrow: 0, borderBottomWidth: 1, borderBottomColor: 'rgba(0,0,0,0.06)' },
  ychip: { paddingHorizontal: 12, paddingVertical: 5, borderRadius: 14, backgroundColor: '#fff', borderWidth: 1, borderColor: 'rgba(0,0,0,0.1)' },
  ychipActive: { backgroundColor: ACCENT, borderColor: ACCENT },
  ychipTxt: { fontSize: 12, color: TEXT },
  ychipTxtActive: { color: '#fff', fontWeight: '700' },

  densityRow: { flexDirection: 'row', alignItems: 'flex-end', paddingHorizontal: PADDING, paddingTop: 8, gap: 8 },
  densityLabel: { fontSize: 10, color: MUTED, marginBottom: 16, width: 36 },
  densityBars: { flex: 1, flexDirection: 'row', alignItems: 'flex-end', justifyContent: 'space-around', height: 44 },
  densityCol: { alignItems: 'center', flex: 1, justifyContent: 'flex-end' },
  densityCount: { fontSize: 7, color: MUTED },
  densityFill: { width: '60%', maxWidth: 14, backgroundColor: BLUE, borderRadius: 2 },
  densityFillCur: { backgroundColor: ACCENT },
  densityTick: { fontSize: 8, color: MUTED, marginTop: 2 },

  main: { flex: 1, backgroundColor: '#0d1117', position: 'relative', minHeight: 180 },
  mainImg: { flex: 1, width: '100%', backgroundColor: '#0d1117' },
  arrow: { position: 'absolute', top: '50%', marginTop: -24, width: 44, height: 48, justifyContent: 'center', alignItems: 'center', backgroundColor: 'rgba(0,0,0,0.35)', borderRadius: 24 },
  arrowLeft: { left: 8 },
  arrowRight: { right: 8 },
  arrowTxt: { color: '#fff', fontSize: 30, fontWeight: '300', lineHeight: 32 },
  dayHeader: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: PADDING, paddingTop: 6, paddingBottom: 2, backgroundColor: '#161b22' },
  dayHeaderTxt: { fontSize: 11, color: '#cbd5e1', fontWeight: '600' },
  overlay: { position: 'absolute', left: 0, right: 0, bottom: 0, padding: 10, backgroundColor: 'rgba(0,0,0,0.45)' },
  dt: { color: '#fff', fontSize: 13, fontWeight: '600' },
  loc: { color: '#cbd5e1', fontSize: 11, marginTop: 2 },

  info: { maxHeight: 180, backgroundColor: '#fff', borderTopWidth: 1, borderTopColor: 'rgba(0,0,0,0.06)' },
  infoTitle: { fontSize: 13, fontWeight: '700', color: TEXT },
  peopleLine: { fontSize: 13, color: ACCENT, marginTop: 4, fontWeight: '600' },
  personCard: { alignItems: 'center', width: 60 },
  personFace: { width: 48, height: 48, borderRadius: 24, backgroundColor: '#eee' },
  personName: { fontSize: 11, color: TEXT, fontWeight: '600', marginTop: 3, maxWidth: 58 },
  personRel: { fontSize: 9, color: MUTED, maxWidth: 58 },
  tags: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginTop: 8 },
  tag: { fontSize: 11, paddingHorizontal: 9, paddingVertical: 3, borderRadius: 12, borderWidth: 1 },
  desc: { fontSize: 12, color: MUTED, marginTop: 8, lineHeight: 17 },

  strip: { maxHeight: 60, flexGrow: 0, backgroundColor: '#161b22' },
  thumb: { width: 48, height: 48, borderRadius: 6, backgroundColor: '#222', opacity: 0.55 },
  thumbActive: { opacity: 1, borderWidth: 2, borderColor: ACCENT },
  counter: { textAlign: 'center', fontSize: 10, color: MUTED, paddingVertical: 3, backgroundColor: '#161b22' },

  modalOverlay: { flex: 1, justifyContent: 'flex-end', backgroundColor: 'rgba(0,0,0,0.4)' },
  modalCard: { backgroundColor: BG, borderTopLeftRadius: 18, borderTopRightRadius: 18, padding: PADDING, maxHeight: '75%' },
  modalHead: { flexDirection: 'row', alignItems: 'center', marginBottom: 8 },
  modalTitle: { fontSize: 20, fontWeight: '800', color: TEXT },
  modalRel: { fontSize: 12, color: ACCENT, marginTop: 2 },
  modalClose: { fontSize: 22, color: TEXT, paddingHorizontal: 6 },
  modalMeta: { fontSize: 13, color: MUTED },
  modalFace: { width: 80, height: 80, borderRadius: 10, backgroundColor: '#eee' },
  modalSection: { marginTop: 12 },
  modalSectionTitle: { fontSize: 13, fontWeight: '700', color: TEXT, marginBottom: 4 },
  modalJson: { fontSize: 11, color: MUTED, fontFamily: 'monospace', lineHeight: 16 },
});
