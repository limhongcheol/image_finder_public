import { useEffect, useState } from 'react';
import {
  View, Text, FlatList, Image, TextInput, TouchableOpacity, Modal, ScrollView,
  ActivityIndicator, StyleSheet, Dimensions,
} from 'react-native';
import { getAllProfiles, getPersonNames, setPersonName, getPerson, getPersonZeroRanking, personImageUrl } from '../api';
import FaceImage from '../components/FaceImage';
import RankingSection from '../components/RankingSection';
import { ACCENT, BLUE, BG, TEXT, MUTED, PADDING } from '../theme';

const FACE = (Dimensions.get('window').width - PADDING * 2 - 24) / 3;

export default function PeopleScreen({ openPersonId, onConsumePerson }) {
  const [loading, setLoading] = useState(true);
  const [profiles, setProfiles] = useState([]);
  const [names, setNames] = useState({});      // { person_id: 이름 }
  const [drafts, setDrafts] = useState({});     // 편집 중 입력값
  const [detail, setDetail] = useState(null);   // 인물 상세 (모달)
  const [detailLoading, setDetailLoading] = useState(false);
  const [ranking, setRanking] = useState(null); // 나 중심 동행 랭킹

  useEffect(() => {
    (async () => {
      const [p, n, rk] = await Promise.all([getAllProfiles(), getPersonNames(), getPersonZeroRanking()]);
      setProfiles(Array.isArray(p) ? p : []);
      setNames(n || {});
      setRanking(rk);
      setLoading(false);
    })();
  }, []);

  const saveName = async (pid) => {
    const name = (drafts[pid] ?? names[pid] ?? '').trim();
    await setPersonName(pid, name);
    setNames(prev => ({ ...prev, [pid]: name }));
  };

  const openDetail = async (pid) => {
    setDetailLoading(true); setDetail({ person_id: pid });
    const d = await getPerson(pid);
    setDetail(d || { person_id: pid, images: [] });
    setDetailLoading(false);
  };

  // 다른 탭(라이프스타일)에서 넘어온 인물 → 자동으로 상세 열기
  useEffect(() => {
    if (openPersonId) { openDetail(openPersonId); onConsumePerson?.(); }
  }, [openPersonId]);

  if (loading) return <View style={styles.center}><ActivityIndicator color={ACCENT} /></View>;

  return (
    <View style={styles.wrap}>
      <Text style={styles.title}>인물</Text>
      <Text style={styles.sub}>{profiles.length}명 · 이름을 지정할 수 있어요</Text>

      {profiles.length === 0 ? (
        <View style={styles.center}><Text style={{ color: MUTED }}>인물 데이터 없음 (knowledge_graph.json 확인)</Text></View>
      ) : (
        <FlatList
          data={profiles}
          keyExtractor={(it, i) => String(it.person_id ?? i)}
          contentContainerStyle={{ paddingVertical: 10, gap: 12 }}
          ListHeaderComponent={<RankingSection data={ranking} onPersonTap={openDetail} />}
          renderItem={({ item }) => {
            const pid = item.person_id;
            const face = (item.images || [])[0];
            return (
              <View style={styles.card}>
                <TouchableOpacity style={styles.head} activeOpacity={0.7} onPress={() => openDetail(pid)}>
                  {face
                    ? <FaceImage uri={personImageUrl(pid, face)} style={styles.face} emojiSize={18} />
                    : <View style={[styles.face, styles.noFace]}><Text style={{ color: MUTED }}>👤</Text></View>}
                  <View style={{ flex: 1 }}>
                    <Text style={styles.rel}>{names[pid] || item.relationship_type || '관계 미상'}</Text>
                    <Text style={styles.meta}>
                      {names[pid] ? `${item.relationship_type || '관계 미상'} · ` : ''}친밀도 {item.intimacy_score ?? '-'} · {item.estimated_age || ''} {item.gender || ''} · 사진 {item.photo_count ?? 0}장
                    </Text>
                  </View>
                  <Text style={styles.chevron}>›</Text>
                </TouchableOpacity>

                {/* 이름 지정 */}
                <View style={styles.nameRow}>
                  <TextInput
                    style={styles.nameInput}
                    value={drafts[pid] ?? names[pid] ?? ''}
                    onChangeText={(t) => setDrafts(prev => ({ ...prev, [pid]: t }))}
                    placeholder="이름 입력"
                    placeholderTextColor={MUTED}
                  />
                  <TouchableOpacity style={styles.saveBtn} onPress={() => saveName(pid)}>
                    <Text style={styles.saveTxt}>저장</Text>
                  </TouchableOpacity>
                </View>

                {!!item.relationship_narrative && (
                  <Text style={styles.narrative} numberOfLines={3}>{item.relationship_narrative}</Text>
                )}
              </View>
            );
          }}
        />
      )}

      {/* 인물 상세 (모달) — /api/person/<id> */}
      <Modal visible={!!detail} animationType="slide" onRequestClose={() => setDetail(null)}>
        <View style={styles.detailWrap}>
          <View style={styles.detailHead}>
            <Text style={styles.detailTitle}>
              {detail && (names[detail.person_id] || detail.person_id)}
            </Text>
            <TouchableOpacity onPress={() => setDetail(null)}><Text style={{ fontSize: 22, color: TEXT }}>✕</Text></TouchableOpacity>
          </View>
          {detailLoading ? (
            <View style={styles.center}><ActivityIndicator color={ACCENT} /></View>
          ) : (
            <ScrollView contentContainerStyle={{ padding: PADDING }}>
              {/* 얼굴 */}
              <View style={styles.faceGrid}>
                {(detail?.images || []).map((fn, i) => (
                  <FaceImage key={i} uri={personImageUrl(detail.person_id, fn)} style={styles.faceImg} />
                ))}
              </View>

              {/* 요약 정보 칩 */}
              <View style={styles.infoChips}>
                {[
                  detail?.level2?.estimated_age && ['나이', String(detail.level2.estimated_age)],
                  detail?.level2?.gender && ['성별', String(detail.level2.gender)],
                  ['사진', `${detail?.photo_count ?? detail?.level2?.photo_count ?? 0}장`],
                  detail?.level3?._distinct_years && ['활동', `${detail.level3._distinct_years}년`],
                ].filter(Boolean).map(([label, value], i) => (
                  <View key={i} style={styles.infoChip}>
                    <Text style={styles.infoChipLabel}>{label}</Text>
                    <Text style={styles.infoChipValue}>{value}</Text>
                  </View>
                ))}
              </View>

              {/* 요약 */}
              {!!detail?.level2?.summary && (
                <View style={styles.infoCard}>
                  <Text style={styles.infoCardTitle}>📝 요약</Text>
                  <Text style={styles.infoCardBody}>{detail.level2.summary}</Text>
                </View>
              )}

              {/* 관계 서사 */}
              {!!detail?.level3?.relationship_narrative && (
                <View style={[styles.infoCard, styles.relCard]}>
                  <Text style={styles.infoCardTitle}>🤝 관계</Text>
                  <Text style={styles.infoCardBody}>{detail.level3.relationship_narrative}</Text>
                </View>
              )}
            </ScrollView>
          )}
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { flex: 1, backgroundColor: BG, paddingHorizontal: PADDING },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: BG, padding: 40 },
  title: { fontSize: 24, fontWeight: '700', color: TEXT, marginTop: 8 },
  sub: { fontSize: 13, color: MUTED, marginTop: 4, marginBottom: 8 },
  card: { backgroundColor: '#fff', borderRadius: 12, padding: 14, borderWidth: 1, borderColor: 'rgba(0,0,0,0.06)' },
  head: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  face: { width: 52, height: 52, borderRadius: 26, backgroundColor: '#eee' },
  noFace: { justifyContent: 'center', alignItems: 'center' },
  rel: { fontSize: 15, fontWeight: '600', color: TEXT },
  meta: { fontSize: 12, color: MUTED, marginTop: 2 },
  nameRow: { flexDirection: 'row', gap: 8, marginTop: 12 },
  nameInput: {
    flex: 1, height: 40, backgroundColor: BG, borderRadius: 8, paddingHorizontal: 12,
    fontSize: 14, color: TEXT, borderWidth: 1, borderColor: 'rgba(0,0,0,0.08)',
  },
  saveBtn: { backgroundColor: BLUE, borderRadius: 8, paddingHorizontal: 16, justifyContent: 'center' },
  saveTxt: { color: '#fff', fontSize: 14, fontWeight: '600' },
  narrative: { fontSize: 12, color: MUTED, marginTop: 10, lineHeight: 18 },
  chevron: { fontSize: 24, color: MUTED, marginLeft: 4 },
  // 인물 상세 모달
  detailWrap: { flex: 1, backgroundColor: BG },
  detailHead: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingHorizontal: PADDING, paddingTop: 48, paddingBottom: 14,
    borderBottomWidth: 1, borderBottomColor: 'rgba(0,0,0,0.06)',
  },
  detailTitle: { fontSize: 20, fontWeight: '700', color: TEXT },
  detailMeta: { fontSize: 13, color: MUTED, marginBottom: 12 },
  faceGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 12 },
  faceImg: { width: FACE, height: FACE, borderRadius: 10, backgroundColor: '#eee' },
  detailSection: { marginTop: 18 },
  detailSectionTitle: { fontSize: 14, fontWeight: '600', color: ACCENT, marginBottom: 6 },
  detailJson: { fontSize: 11, color: MUTED, fontFamily: 'monospace' },

  infoChips: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginTop: 16 },
  infoChip: { backgroundColor: '#fff', borderRadius: 10, paddingVertical: 8, paddingHorizontal: 14, borderWidth: 1, borderColor: 'rgba(0,0,0,0.08)', alignItems: 'center' },
  infoChipLabel: { fontSize: 10, color: MUTED },
  infoChipValue: { fontSize: 14, fontWeight: '700', color: TEXT, marginTop: 1 },
  infoCard: { backgroundColor: '#fff', borderRadius: 12, padding: 14, marginTop: 14, borderWidth: 1, borderColor: 'rgba(0,0,0,0.06)' },
  relCard: { backgroundColor: 'rgba(200,90,48,0.05)', borderColor: 'rgba(200,90,48,0.2)' },
  infoCardTitle: { fontSize: 14, fontWeight: '700', color: TEXT, marginBottom: 6 },
  infoCardBody: { fontSize: 13, color: TEXT, lineHeight: 20 },
});
