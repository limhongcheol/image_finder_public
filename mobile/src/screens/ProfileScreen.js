import { useEffect, useState } from 'react';
import { View, Text, ScrollView, ActivityIndicator, StyleSheet } from 'react-native';
import { getSummary, getLifestyleStats, getYearly } from '../api';
import { ACCENT, BLUE, BG, TEXT, MUTED, PADDING } from '../theme';

// 나의 프로필: 페르소나 + 음식/풍경/사회성/활동 통계 + 연도별 (web_app "나의 프로필")
export default function ProfileScreen() {
  const [loading, setLoading] = useState(true);
  const [summary, setSummary] = useState(null);
  const [stats, setStats] = useState(null);
  const [yearly, setYearly] = useState(null);

  useEffect(() => {
    (async () => {
      const [s, st, y] = await Promise.all([getSummary(), getLifestyleStats(), getYearly()]);
      setSummary(s); setStats(st); setYearly(y);
      setLoading(false);
    })();
  }, []);

  if (loading) return <View style={styles.center}><ActivityIndicator color={ACCENT} /></View>;

  const food = stats?.food || {};
  const land = stats?.landscape || {};
  const social = stats?.social || {};
  const act = stats?.activity || {};
  const intensity = act.intensity_distribution || {};
  const categories = Object.entries(act.category_distribution || {}).sort((a, b) => b[1] - a[1]).slice(0, 6);
  const cuisines = (food.top_cuisines || []).slice(0, 6);
  const yearEntries = yearly ? Object.entries(yearly).sort((a, b) => b[0].localeCompare(a[0])) : [];
  const pct = (x) => Math.round((x || 0) * 100);

  return (
    <ScrollView style={styles.wrap} contentContainerStyle={{ paddingBottom: 40 }}>
      <Text style={styles.title}>나의 프로필</Text>
      <Text style={styles.sub}>사진 {stats?.total_photos ?? 0}장 분석</Text>

      {summary && (
        <View style={[styles.card, styles.personaCard]}>
          <Text style={styles.cardTitle}>🧬 나의 페르소나</Text>
          {!!summary.overall_persona && <Text style={styles.persona}>{summary.overall_persona}</Text>}
          {!!(summary.highlight_tags?.length) && (
            <View style={styles.chips}>
              {summary.highlight_tags.map((t, i) => <Text key={i} style={styles.chip}>{t}</Text>)}
            </View>
          )}
        </View>
      )}

      {cuisines.length > 0 && (
        <Section title="🍽️ 음식" sub={`전체의 ${pct(food.ratio)}% · ${food.photo_count || 0}장`}>
          {cuisines.map(([name, count], i) => (
            <Bar key={i} label={name} value={count} max={cuisines[0][1]} color={ACCENT} suffix="장" />
          ))}
        </Section>
      )}

      {(land.sea_score != null) && (
        <Section title="🌄 풍경" sub={`전체의 ${pct(land.ratio)}% · ${land.photo_count || 0}장`}>
          <Bar label="바다" value={land.sea_score} max={Math.max(land.sea_score, land.mountain_score, land.city_score)} color="#06b6d4" />
          <Bar label="산" value={land.mountain_score} max={Math.max(land.sea_score, land.mountain_score, land.city_score)} color="#22c55e" />
          <Bar label="도시" value={land.city_score} max={Math.max(land.sea_score, land.mountain_score, land.city_score)} color="#f59e0b" />
        </Section>
      )}

      {(social.solo_ratio != null) && (
        <Section title="👥 사회성" sub={`평균 ${social.avg_people_per_photo?.toFixed?.(1) ?? '-'}명/사진`}>
          <Bar label="혼자" value={pct(social.solo_ratio)} max={100} color={BLUE} suffix="%" />
          <Bar label="함께" value={pct(social.with_others_ratio)} max={100} color={ACCENT} suffix="%" />
        </Section>
      )}

      {(categories.length > 0 || Object.keys(intensity).length > 0) && (
        <Section title="🏃 활동">
          {Object.keys(intensity).length > 0 && (
            <>
              <Text style={styles.subLabel}>강도</Text>
              <Bar label="낮음" value={intensity.low || 0} max={Math.max(1, ...Object.values(intensity))} color="#6b7280" suffix="장" />
              <Bar label="보통" value={intensity.medium || 0} max={Math.max(1, ...Object.values(intensity))} color="#f59e0b" suffix="장" />
              <Bar label="높음" value={intensity.high || 0} max={Math.max(1, ...Object.values(intensity))} color="#ef4444" suffix="장" />
            </>
          )}
          {categories.length > 0 && (
            <>
              <Text style={styles.subLabel}>분류</Text>
              {categories.map(([name, count], i) => (
                <Bar key={i} label={name} value={count} max={categories[0][1]} color={BLUE} suffix="장" />
              ))}
            </>
          )}
        </Section>
      )}

      {yearEntries.length > 0 && (
        <View style={{ marginTop: 6 }}>
          <Text style={styles.sectionHeader}>📅 연도별 이야기</Text>
          {yearEntries.map(([year, y]) => (
            <View key={year} style={[styles.card, styles.yearCard]}>
              <Text style={styles.yearNum}>{year}</Text>
              {!!y.headline && <Text style={styles.yearHeadline}>{y.headline}</Text>}
              {!!y.summary && <Text style={styles.yearSummary}>{y.summary}</Text>}
              {!!(y.key_themes?.length) && (
                <View style={styles.chips}>
                  {y.key_themes.map((t, i) => <Text key={i} style={[styles.chip, styles.themeChip]}>{t}</Text>)}
                </View>
              )}
              {!!y.highlight && <Text style={styles.yearHighlight}>✨ {y.highlight}</Text>}
            </View>
          ))}
        </View>
      )}

      {!summary && !stats && !yearly && <Empty text="프로필 데이터 없음 (백엔드/파일 확인)" />}
    </ScrollView>
  );
}

const Section = ({ title, sub, children }) => (
  <View style={styles.card}>
    <View style={styles.cardHead}>
      <Text style={styles.cardTitle}>{title}</Text>
      {!!sub && <Text style={styles.cardSub}>{sub}</Text>}
    </View>
    {children}
  </View>
);

const Bar = ({ label, value, max, color, suffix = '' }) => (
  <View style={styles.barRow}>
    <Text style={styles.barLabel} numberOfLines={1}>{label}</Text>
    <View style={styles.barTrack}>
      <View style={[styles.barFill, { width: `${Math.min(100, (value / (max || 1)) * 100)}%`, backgroundColor: color }]} />
    </View>
    <Text style={styles.barVal}>{value}{suffix}</Text>
  </View>
);

const Empty = ({ text }) => <View style={styles.center}><Text style={{ color: MUTED }}>{text}</Text></View>;

const styles = StyleSheet.create({
  wrap: { flex: 1, backgroundColor: BG, paddingHorizontal: PADDING },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: BG, padding: 40 },
  title: { fontSize: 24, fontWeight: '700', color: TEXT, marginTop: 8 },
  sub: { fontSize: 13, color: MUTED, marginTop: 4, marginBottom: 8 },

  card: { backgroundColor: '#fff', borderRadius: 12, padding: 14, marginTop: 12, borderWidth: 1, borderColor: 'rgba(0,0,0,0.06)' },
  cardHead: { flexDirection: 'row', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: 10 },
  cardTitle: { fontSize: 15, fontWeight: '700', color: TEXT },
  cardSub: { fontSize: 11, color: MUTED },
  subLabel: { fontSize: 11, color: MUTED, marginTop: 8, marginBottom: 2, fontWeight: '600' },

  personaCard: { backgroundColor: 'rgba(200,90,48,0.06)', borderColor: 'rgba(200,90,48,0.25)' },
  persona: { fontSize: 13, color: TEXT, lineHeight: 20, marginBottom: 8 },
  chips: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginTop: 6 },
  chip: { fontSize: 11, color: ACCENT, backgroundColor: 'rgba(200,90,48,0.1)', borderRadius: 12, paddingHorizontal: 10, paddingVertical: 3 },
  themeChip: { color: BLUE, backgroundColor: 'rgba(58,124,198,0.1)' },

  barRow: { flexDirection: 'row', alignItems: 'center', marginVertical: 3, gap: 8 },
  barLabel: { width: 70, fontSize: 12, color: TEXT },
  barTrack: { flex: 1, height: 7, backgroundColor: '#eee', borderRadius: 4, overflow: 'hidden' },
  barFill: { height: 7, borderRadius: 4 },
  barVal: { width: 44, fontSize: 11, color: MUTED, textAlign: 'right' },

  sectionHeader: { fontSize: 16, fontWeight: '700', color: TEXT, marginTop: 18, marginBottom: 2 },
  yearCard: { backgroundColor: 'rgba(58,124,198,0.05)', borderColor: 'rgba(58,124,198,0.2)' },
  yearNum: { fontSize: 18, fontWeight: '800', color: BLUE },
  yearHeadline: { fontSize: 14, fontWeight: '600', color: TEXT, marginTop: 4 },
  yearSummary: { fontSize: 12, color: MUTED, lineHeight: 18, marginTop: 4 },
  yearHighlight: { fontSize: 12, color: ACCENT, marginTop: 8 },
});
