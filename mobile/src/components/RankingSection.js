import { View, Text, Image, TouchableOpacity, StyleSheet } from 'react-native';
import { mediaUrl } from '../api';
import { ACCENT, BLUE, BG, TEXT, MUTED, PADDING } from '../theme';

const MEDALS = ['🥇', '🥈', '🥉'];

// 나(person_0) 중심 동행 랭킹 섹션. data = /api/person_zero_ranking 응답.
export default function RankingSection({ data, onPersonTap }) {
  if (!data || !data.summary) return null;
  const { summary, ranking = [], distribution = [], intimacy_top = [], yearly_best = [] } = data;
  const maxCount = ranking[0]?.count || 1;
  const maxDist = Math.max(1, ...distribution.map(d => d.photos));

  const Face = ({ uri, size = 40 }) =>
    uri
      ? <Image source={{ uri: mediaUrl(uri) }} style={{ width: size, height: size, borderRadius: size / 2, backgroundColor: '#eee' }} />
      : <View style={{ width: size, height: size, borderRadius: size / 2, backgroundColor: '#e3e1dd' }} />;

  return (
    <View>
      {/* 요약 통계 */}
      <View style={styles.statsRow}>
        {[
          { n: summary.my_photos, l: '내 사진' },
          { n: summary.photographed_with, l: '함께한 인물' },
          { n: summary.solo_photos, l: '혼자 사진' },
          { n: summary.total_persons, l: '전체 인물' },
        ].map((s, i) => (
          <View key={i} style={styles.statBox}>
            <Text style={styles.statN}>{s.n ?? 0}</Text>
            <Text style={styles.statL}>{s.l}</Text>
          </View>
        ))}
      </View>

      {/* 함께 많이 찍은 사람 */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>📸 함께 많이 찍은 사람</Text>
        {ranking.slice(0, 10).map((r, i) => (
          <TouchableOpacity key={r.person_id} style={styles.rankRow} activeOpacity={0.7} onPress={() => onPersonTap?.(r.person_id)}>
            <Text style={styles.medal}>{MEDALS[i] || `${i + 1}`}</Text>
            <Face uri={r.thumbnail} />
            <View style={{ flex: 1, marginLeft: 10 }}>
              <Text style={styles.rankName} numberOfLines={1}>{r.relationship_type} · {r.person_id}</Text>
              <View style={styles.barTrack}><View style={[styles.barFill, { width: `${(r.count / maxCount) * 100}%` }]} /></View>
              <Text style={styles.rankMeta}>{r.count}장{r.first_date ? ` · ${r.first_date}~${r.last_date}` : ''}</Text>
            </View>
          </TouchableOpacity>
        ))}
      </View>

      {/* 관계 분포 */}
      {distribution.length > 0 && (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>🧩 관계 분포</Text>
          {distribution.map(d => (
            <View key={d.relationship_type} style={styles.distRow}>
              <Text style={styles.distLabel}>{d.relationship_type}</Text>
              <View style={styles.distBarTrack}><View style={[styles.distBarFill, { width: `${(d.photos / maxDist) * 100}%` }]} /></View>
              <Text style={styles.distVal}>{d.persons}명 · {d.photos}장</Text>
            </View>
          ))}
        </View>
      )}

      {/* 친밀도 TOP */}
      {intimacy_top.length > 0 && (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>💛 친밀도 TOP 5</Text>
          <View style={styles.chipRow}>
            {intimacy_top.map(r => (
              <TouchableOpacity key={r.person_id} style={styles.chip} onPress={() => onPersonTap?.(r.person_id)}>
                <Face uri={r.thumbnail} size={34} />
                <Text style={styles.chipTxt} numberOfLines={1}>{r.person_id}</Text>
                <Text style={styles.chipScore}>{r.intimacy_score}</Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>
      )}

      {/* 연도별 최다 동행 */}
      {yearly_best.length > 0 && (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>📅 연도별 최다 동행</Text>
          {yearly_best.map(y => (
            <TouchableOpacity key={y.year} style={styles.yearRow} activeOpacity={0.7} onPress={() => onPersonTap?.(y.person_id)}>
              <Text style={styles.year}>{y.year}</Text>
              <Face uri={y.thumbnail} size={32} />
              <Text style={styles.yearName} numberOfLines={1}>{y.relationship_type} · {y.person_id}</Text>
              <Text style={styles.yearCnt}>{y.count}장</Text>
            </TouchableOpacity>
          ))}
        </View>
      )}

      <Text style={styles.listHeader}>전체 인물</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  statsRow: { flexDirection: 'row', gap: 8, marginTop: 4, marginBottom: 4 },
  statBox: { flex: 1, backgroundColor: '#fff', borderRadius: 10, paddingVertical: 12, alignItems: 'center', borderWidth: 1, borderColor: 'rgba(0,0,0,0.06)' },
  statN: { fontSize: 18, fontWeight: '800', color: ACCENT },
  statL: { fontSize: 10, color: MUTED, marginTop: 2 },

  card: { backgroundColor: '#fff', borderRadius: 12, padding: 14, marginTop: 12, borderWidth: 1, borderColor: 'rgba(0,0,0,0.06)' },
  cardTitle: { fontSize: 15, fontWeight: '700', color: TEXT, marginBottom: 10 },

  rankRow: { flexDirection: 'row', alignItems: 'center', paddingVertical: 6 },
  medal: { width: 24, fontSize: 15, textAlign: 'center', color: MUTED, fontWeight: '700' },
  rankName: { fontSize: 13, color: TEXT, fontWeight: '600' },
  barTrack: { height: 6, backgroundColor: '#eee', borderRadius: 3, marginVertical: 4, overflow: 'hidden' },
  barFill: { height: 6, backgroundColor: ACCENT },
  rankMeta: { fontSize: 10, color: MUTED },

  distRow: { flexDirection: 'row', alignItems: 'center', paddingVertical: 5, gap: 8 },
  distLabel: { width: 72, fontSize: 12, color: TEXT },
  distBarTrack: { flex: 1, height: 8, backgroundColor: '#eee', borderRadius: 4, overflow: 'hidden' },
  distBarFill: { height: 8, backgroundColor: BLUE },
  distVal: { width: 80, fontSize: 11, color: MUTED, textAlign: 'right' },

  chipRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  chip: { alignItems: 'center', width: 60 },
  chipTxt: { fontSize: 10, color: TEXT, marginTop: 4 },
  chipScore: { fontSize: 11, color: ACCENT, fontWeight: '700' },

  yearRow: { flexDirection: 'row', alignItems: 'center', paddingVertical: 6, gap: 10 },
  year: { width: 44, fontSize: 13, fontWeight: '700', color: BLUE },
  yearName: { flex: 1, fontSize: 13, color: TEXT },
  yearCnt: { fontSize: 12, color: MUTED },

  listHeader: { fontSize: 16, fontWeight: '700', color: TEXT, marginTop: 18, marginBottom: 4 },
});
