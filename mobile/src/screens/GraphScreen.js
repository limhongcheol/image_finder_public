import { useEffect, useRef, useState } from 'react';
import {
  View, Text, ActivityIndicator, Modal, ScrollView, Image,
  TouchableOpacity, Dimensions, StyleSheet,
} from 'react-native';
import { WebView } from 'react-native-webview';
import { getGraph, getPerson, getPersonNames, personImageUrl } from '../api';
import FaceImage from '../components/FaceImage';
import { log } from '../logger';
import { ACCENT, BLUE, BG, TEXT, MUTED, PADDING } from '../theme';

const FACE_SIZE = (Dimensions.get('window').width - PADDING * 2 - 36) / 4;

const RELATION_COLORS = {
  '가족': '#c98282',
  '학창시절친구': '#7aa2f7',
  '직장동료': '#8fa7b8',
  '지인': '#9fb18d',
  '알수없음': '#6f7480',
};

// 자체 캔버스 포스그래프 (외부 CDN 없음 → 오프라인 완전 동작). 핀치줌/팬/탭 지원.
// nodes 는 vis 형식(id, label, color.background/border), edges 는 {from,to}.
function buildHtml(nodes, edges) {
  return `<!DOCTYPE html><html><head>
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<style>html,body{height:100%;margin:0;background:#171a21;overflow:hidden}#c{display:block;touch-action:none}</style>
</head><body><canvas id="c"></canvas><script>
var NODES=${JSON.stringify(nodes)}, EDGES=${JSON.stringify(edges)};
var cv=document.getElementById('c'), ctx=cv.getContext('2d'), DPR=window.devicePixelRatio||1, W=0,H=0;
function resize(){W=window.innerWidth;H=window.innerHeight;cv.width=W*DPR;cv.height=H*DPR;cv.style.width=W+'px';cv.style.height=H+'px';ctx.setTransform(DPR,0,0,DPR,0,0);}
resize();
var N=NODES.length, idx={};
NODES.forEach(function(n,i){var a=i/N*6.2832; n.x=Math.cos(a)*200+(Math.random()-0.5)*40; n.y=Math.sin(a)*200+(Math.random()-0.5)*40; n.vx=0; n.vy=0; idx[n.id]=i;});
function step(){
  for(var i=0;i<N;i++){var a=NODES[i];
    for(var j=i+1;j<N;j++){var b=NODES[j];var dx=a.x-b.x,dy=a.y-b.y,d2=dx*dx+dy*dy+0.01,d=Math.sqrt(d2),f=2200/d2,fx=dx/d*f,fy=dy/d*f;a.vx+=fx;a.vy+=fy;b.vx-=fx;b.vy-=fy;}}
  EDGES.forEach(function(e){var a=NODES[idx[e.from]],b=NODES[idx[e.to]];if(!a||!b)return;var dx=b.x-a.x,dy=b.y-a.y,d=Math.sqrt(dx*dx+dy*dy)+0.01,f=(d-95)*0.02,fx=dx/d*f,fy=dy/d*f;a.vx+=fx;a.vy+=fy;b.vx-=fx;b.vy-=fy;});
  for(var i=0;i<N;i++){var n=NODES[i];n.vx-=n.x*0.002;n.vy-=n.y*0.002;n.x+=n.vx;n.y+=n.vy;n.vx*=0.85;n.vy*=0.85;}
}
for(var s=0;s<320;s++) step();
var minX=1e9,minY=1e9,maxX=-1e9,maxY=-1e9;
NODES.forEach(function(n){minX=Math.min(minX,n.x);minY=Math.min(minY,n.y);maxX=Math.max(maxX,n.x);maxY=Math.max(maxY,n.y);});
var gw=(maxX-minX)||1,gh=(maxY-minY)||1,sc=Math.min((W-80)/gw,(H-80)/gh);sc=Math.min(sc,1.4);
var view={s:sc,tx:(W-gw*sc)/2-minX*sc,ty:(H-gh*sc)/2-minY*sc};
function TS(n){return {x:n.x*view.s+view.tx,y:n.y*view.s+view.ty};}
function draw(){
  ctx.clearRect(0,0,W,H);
  ctx.strokeStyle='rgba(255,255,255,0.13)';ctx.lineWidth=1;
  EDGES.forEach(function(e){var a=NODES[idx[e.from]],b=NODES[idx[e.to]];if(!a||!b)return;var pa=TS(a),pb=TS(b);ctx.beginPath();ctx.moveTo(pa.x,pa.y);ctx.lineTo(pb.x,pb.y);ctx.stroke();});
  NODES.forEach(function(n){var p=TS(n),r=(n.id==='person_0'?13:8),col=(n.color&&n.color.background)||'#7aa2f7',bd=(n.color&&n.color.border)||'#fff';
    ctx.beginPath();ctx.arc(p.x,p.y,r,0,6.2832);ctx.fillStyle=col;ctx.fill();ctx.lineWidth=2;ctx.strokeStyle=bd;ctx.stroke();
    ctx.fillStyle='#c9d1d9';ctx.font='10px sans-serif';ctx.textAlign='center';ctx.fillText(String(n.label||'').slice(0,8),p.x,p.y-r-3);});
}
draw();
function send(o){try{window.ReactNativeWebView.postMessage(JSON.stringify(o));}catch(e){}}
function d2(t){var dx=t[0].clientX-t[1].clientX,dy=t[0].clientY-t[1].clientY;return Math.sqrt(dx*dx+dy*dy);}
function hit(sx,sy){var best=null,bd=1e9;NODES.forEach(function(n){var p=TS(n),d=(p.x-sx)*(p.x-sx)+(p.y-sy)*(p.y-sy);if(d<bd){bd=d;best=n;}});if(best&&bd<500)send({type:'nodeClick',personId:best.id});}
var drag=null,ld=0;
cv.addEventListener('touchstart',function(ev){if(ev.touches.length===1)drag={x:ev.touches[0].clientX,y:ev.touches[0].clientY,moved:false,t:Date.now()};else if(ev.touches.length===2){ld=d2(ev.touches);drag=null;}});
cv.addEventListener('touchmove',function(ev){ev.preventDefault();
  if(ev.touches.length===2){var d=d2(ev.touches);if(ld){var f=d/ld,cx=(ev.touches[0].clientX+ev.touches[1].clientX)/2,cy=(ev.touches[0].clientY+ev.touches[1].clientY)/2;view.tx=cx-(cx-view.tx)*f;view.ty=cy-(cy-view.ty)*f;view.s*=f;}ld=d;draw();}
  else if(ev.touches.length===1&&drag){var t=ev.touches[0],dx=t.clientX-drag.x,dy=t.clientY-drag.y;if(Math.abs(dx)+Math.abs(dy)>5)drag.moved=true;view.tx+=dx;view.ty+=dy;drag.x=t.clientX;drag.y=t.clientY;draw();}
},{passive:false});
cv.addEventListener('touchend',function(ev){if(drag&&!drag.moved&&Date.now()-drag.t<400)hit(drag.x,drag.y);drag=null;ld=0;});
window.addEventListener('resize',function(){resize();draw();});
send({type:'debug',msg:'graph ready (canvas), nodes='+N});
</script></body></html>`;
}

export default function GraphScreen() {
  const [loading, setLoading] = useState(true);
  const [html, setHtml] = useState(null);
  const [count, setCount] = useState(0);
  const [error, setError] = useState(false);
  const [names, setNames] = useState({});

  // 인물 상세 모달
  const [detail, setDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);

  useEffect(() => {
    (async () => {
      const [data, n] = await Promise.all([getGraph(), getPersonNames()]);
      setNames(n || {});
      log.info('graph', `getGraph → nodes=${data?.nodes?.length ?? 'null'} edges=${data?.edges?.length ?? 0}`);
      if (!data || !data.nodes) { setError(true); setLoading(false); return; }
      setCount(Math.max(0, (data.nodes.length || 1) - 1));   // 나(person_0) 제외
      setHtml(buildHtml(data.nodes, data.edges || []));
      setLoading(false);
    })();
  }, []);

  // WebView에서 노드 클릭 메시지 수신
  const onMessage = async (event) => {
    const raw = event?.nativeEvent?.data;
    log.info('graph', `WebView 메시지 수신: ${raw}`);
    try {
      const msg = JSON.parse(raw);
      if (msg.type === 'nodeClick' && msg.personId) {
        openDetail(msg.personId);
      }
    } catch (e) {
      log.warn('graph', `메시지 파싱 실패: ${e.message}`);
    }
  };

  const openDetail = async (personId) => {
    log.info('graph', `상세 열기 → ${personId}`);
    setDetailLoading(true);
    setDetail({ person_id: personId });
    const d = await getPerson(personId);
    if (!d) log.warn('graph', `getPerson(${personId}) 응답 없음 (백엔드 404/네트워크?)`);
    setDetail(d || { person_id: personId, images: [] });
    setDetailLoading(false);
  };

  const displayName = (pid) => names[pid] || pid;

  // ── 인물 상세 모달 콘텐츠 렌더 ──
  const renderDetail = () => {
    if (!detail) return null;
    const isMe = detail.person_id === 'person_0';
    const l2 = detail.level2 || {};
    const l3 = detail.level3 || {};
    const rel = l3.relationship_type || '알수없음';
    const relColor = RELATION_COLORS[rel] || '#6f7480';
    const intimacy = parseInt(l3.intimacy_score || 0) || 0;
    const narr = l3.relationship_narrative || '';
    const basis = l3.intimacy_basis || '';
    const ageReas = l3.age_based_reasoning || '';
    const keywords = [...(l3.episode_keywords || []), ...((l3.knowledge_graph_edge || {}).tags || [])];
    const lifestyle = l2.lifestyle || {};
    const summary = l2.summary || '';
    const age = l2.estimated_age || '';
    const gender = l2.gender || '';
    const images = detail.images || [];
    const mc = l3.meeting_context || {};
    const temporal = l2.temporal_patterns || {};

    return (
      <ScrollView contentContainerStyle={{ padding: PADDING, paddingBottom: 40 }}>
        {/* ── 헤더 영역 ── */}
        <View style={styles.detailHeader}>
          {isMe ? (
            <Text style={[styles.detailName, { color: '#FFD700' }]}>⭐ 나 (person_0)</Text>
          ) : (
            <Text style={styles.detailName}>{displayName(detail.person_id)}</Text>
          )}
          {!isMe && (
            <Text style={styles.detailSubName}>{detail.person_id}</Text>
          )}
        </View>

        {/* ── 배지 ── */}
        <View style={styles.badges}>
          {!isMe && (
            <View style={[styles.badge, { backgroundColor: relColor + '22', borderColor: relColor + '66' }]}>
              <Text style={[styles.badgeText, { color: relColor }]}>{rel}</Text>
            </View>
          )}
          {isMe && (
            <View style={[styles.badge, { backgroundColor: '#FFD70022', borderColor: '#FFD70066' }]}>
              <Text style={[styles.badgeText, { color: '#FFD700' }]}>기준 노드</Text>
            </View>
          )}
          {!!age && (
            <View style={[styles.badge, { backgroundColor: '#4ECDC422', borderColor: '#4ECDC466' }]}>
              <Text style={[styles.badgeText, { color: '#4ECDC4' }]}>{age}</Text>
            </View>
          )}
          {!!gender && (
            <View style={[styles.badge, { backgroundColor: '#7c6af722', borderColor: '#7c6af766' }]}>
              <Text style={[styles.badgeText, { color: '#7c6af7' }]}>{gender}</Text>
            </View>
          )}
        </View>

        {/* ── 통계 카드 ── */}
        <View style={styles.statsRow}>
          <View style={styles.statCard}>
            <Text style={styles.statVal}>{detail.photo_count ?? 0}</Text>
            <Text style={styles.statLabel}>사진 수</Text>
          </View>
          {!isMe && (
            <View style={styles.statCard}>
              <Text style={[styles.statVal, { color: relColor }]}>{intimacy}</Text>
              <Text style={styles.statLabel}>친밀도/10</Text>
            </View>
          )}
          <View style={styles.statCard}>
            <Text style={styles.statVal}>{(lifestyle.activity_keywords || []).length}</Text>
            <Text style={styles.statLabel}>활동 유형</Text>
          </View>
        </View>

        {/* ── 친밀도 게이지 (나가 아닐 때) ── */}
        {!isMe && intimacy > 0 && (
          <View style={styles.infoCard}>
            <View style={styles.gaugeHeader}>
              <Text style={styles.infoCardTitle}>친밀도 게이지</Text>
              <Text style={[styles.gaugeVal, { color: relColor }]}>{intimacy}/10</Text>
            </View>
            <View style={styles.gaugeTrack}>
              <View style={[styles.gaugeFill, { width: `${intimacy * 10}%` }]} />
            </View>
          </View>
        )}

        {/* ── 얼굴 사진 ── */}
        {images.length > 0 && (
          <View style={styles.infoCard}>
            <Text style={styles.infoCardTitle}>📸 얼굴 사진</Text>
            <View style={styles.faceGrid}>
              {images.map((fn, i) => (
                <FaceImage key={i} uri={personImageUrl(detail.person_id, fn)} style={styles.faceImg} />
              ))}
            </View>
          </View>
        )}

        {/* ── 관계 서사 ── */}
        {!!narr && (
          <View style={[styles.infoCard, styles.narrCard]}>
            <Text style={styles.infoCardTitle}>💬 관계 서사</Text>
            <Text style={styles.infoCardBody}>{narr}</Text>
          </View>
        )}

        {/* ── 나이 기반 추론 ── */}
        {!!ageReas && (
          <View style={styles.infoCard}>
            <Text style={styles.infoCardTitle}>📊 나이 기반 추론</Text>
            <Text style={styles.infoCardBody}>{ageReas}</Text>
          </View>
        )}

        {/* ── 친밀도 근거 ── */}
        {!!basis && (
          <View style={styles.infoCard}>
            <Text style={styles.infoCardTitle}>⚖️ 친밀도 근거</Text>
            <Text style={styles.infoCardBody}>{basis}</Text>
          </View>
        )}

        {/* ── 키워드 ── */}
        {keywords.length > 0 && (
          <View style={styles.infoCard}>
            <Text style={styles.infoCardTitle}>🏷️ 키워드</Text>
            <View style={styles.tagsWrap}>
              {keywords.map((kw, i) => (
                <View key={i} style={styles.kwTag}>
                  <Text style={styles.kwTagText}>{kw}</Text>
                </View>
              ))}
            </View>
          </View>
        )}

        {/* ── 주요 만남 ── */}
        {!!mc.primary_setting && (
          <View style={styles.infoCard}>
            <Text style={styles.infoCardTitle}>📍 주요 만남</Text>
            <Text style={styles.infoCardBody}>{mc.primary_setting}</Text>
            {!!temporal.activity_frequency && (
              <Text style={styles.smallMuted}>빈도: {temporal.activity_frequency}</Text>
            )}
          </View>
        )}

        {/* ── 라이프스타일 요약 ── */}
        {!!summary && (
          <View style={styles.infoCard}>
            <Text style={styles.infoCardTitle}>📈 라이프스타일</Text>
            <Text style={styles.infoCardBody}>{summary}</Text>
            {(lifestyle.preferred_location_types || []).length > 0 && (
              <View style={[styles.tagsWrap, { marginTop: 8 }]}>
                {lifestyle.preferred_location_types.map((loc, i) => (
                  <View key={`loc-${i}`} style={[styles.kwTag, styles.locTag]}>
                    <Text style={[styles.kwTagText, { color: BLUE }]}>📍 {loc}</Text>
                  </View>
                ))}
              </View>
            )}
            {(lifestyle.activity_keywords || []).length > 0 && (
              <View style={[styles.tagsWrap, { marginTop: 6 }]}>
                {lifestyle.activity_keywords.map((act, i) => (
                  <View key={`act-${i}`} style={[styles.kwTag, styles.actTag]}>
                    <Text style={[styles.kwTagText, { color: ACCENT }]}>⚡ {act}</Text>
                  </View>
                ))}
              </View>
            )}
          </View>
        )}

        {/* ── 나(person_0)일 때: 프로필 요약 ── */}
        {isMe && !!detail.profile?.profile_summary && (
          <View style={styles.infoCard}>
            <Text style={styles.infoCardTitle}>🪪 나의 프로필</Text>
            <Text style={styles.infoCardBody}>{detail.profile.profile_summary}</Text>
          </View>
        )}

        {/* ── 나(person_0)일 때: 컨텍스트 확장 ── */}
        {isMe && !!detail.context_expansion?.summary && (
          <View style={styles.infoCard}>
            <Text style={styles.infoCardTitle}>🗺️ 컨텍스트 확장</Text>
            <Text style={styles.infoCardBody}>{detail.context_expansion.summary}</Text>
          </View>
        )}
      </ScrollView>
    );
  };

  return (
    <View style={styles.wrap}>
      <View style={styles.head}>
        <Text style={styles.title}>관계 그래프</Text>
        <Text style={styles.sub}>나와 연결된 인물 {count}명 · 노드를 탭하면 상세 보기</Text>
      </View>
      {loading ? (
        <View style={styles.center}><ActivityIndicator color={ACCENT} /></View>
      ) : error ? (
        <View style={styles.center}><Text style={{ color: MUTED }}>관계 데이터 없음 (knowledge_graph.json 확인)</Text></View>
      ) : (
        <WebView
          originWhitelist={['*']}
          source={{ html }}
          style={styles.web}
          javaScriptEnabled
          domStorageEnabled
          mixedContentMode="always"
          onMessage={onMessage}
        />
      )}

      {/* ── 인물 상세 모달 ── */}
      <Modal visible={!!detail} animationType="slide" onRequestClose={() => setDetail(null)}>
        <View style={styles.modalWrap}>
          <View style={styles.modalHead}>
            <Text style={styles.modalTitle}>
              {detail && (names[detail.person_id] || detail.person_id)}
            </Text>
            <TouchableOpacity onPress={() => setDetail(null)} style={styles.closeBtn}>
              <Text style={styles.closeTxt}>✕</Text>
            </TouchableOpacity>
          </View>
          {detailLoading ? (
            <View style={styles.center}><ActivityIndicator color={ACCENT} /></View>
          ) : (
            renderDetail()
          )}
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { flex: 1, backgroundColor: BG },
  head: { paddingHorizontal: PADDING, paddingTop: 8, paddingBottom: 8 },
  title: { fontSize: 24, fontWeight: '700', color: TEXT },
  sub: { fontSize: 13, color: MUTED, marginTop: 4 },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  web: { flex: 1, backgroundColor: '#171a21' },

  // 모달
  modalWrap: { flex: 1, backgroundColor: BG },
  modalHead: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingHorizontal: PADDING, paddingTop: 48, paddingBottom: 14,
    borderBottomWidth: 1, borderBottomColor: 'rgba(0,0,0,0.06)',
  },
  modalTitle: { fontSize: 20, fontWeight: '700', color: TEXT },
  closeBtn: { padding: 4 },
  closeTxt: { fontSize: 22, color: TEXT },

  // 상세 헤더
  detailHeader: { marginBottom: 10 },
  detailName: { fontSize: 22, fontWeight: '800', color: TEXT },
  detailSubName: { fontSize: 12, color: MUTED, marginTop: 2 },

  // 배지
  badges: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginBottom: 14 },
  badge: {
    paddingHorizontal: 10, paddingVertical: 3, borderRadius: 12,
    borderWidth: 1,
  },
  badgeText: { fontSize: 12, fontWeight: '600' },

  // 통계
  statsRow: { flexDirection: 'row', gap: 8, marginBottom: 14 },
  statCard: {
    flex: 1, backgroundColor: '#fff', borderRadius: 12, padding: 12,
    alignItems: 'center', borderWidth: 1, borderColor: 'rgba(0,0,0,0.06)',
  },
  statVal: { fontSize: 22, fontWeight: '800', color: ACCENT },
  statLabel: { fontSize: 10, color: MUTED, marginTop: 2 },

  // 친밀도 게이지
  gaugeHeader: {
    flexDirection: 'row', justifyContent: 'space-between',
    alignItems: 'center', marginBottom: 8,
  },
  gaugeVal: { fontSize: 14, fontWeight: '700' },
  gaugeTrack: {
    height: 8, backgroundColor: '#eee', borderRadius: 4, overflow: 'hidden',
  },
  gaugeFill: {
    height: 8, borderRadius: 4,
    backgroundColor: ACCENT,
  },

  // 정보 카드
  infoCard: {
    backgroundColor: '#fff', borderRadius: 12, padding: 14,
    marginBottom: 12, borderWidth: 1, borderColor: 'rgba(0,0,0,0.06)',
  },
  narrCard: {
    backgroundColor: 'rgba(200,90,48,0.05)', borderColor: 'rgba(200,90,48,0.2)',
  },
  infoCardTitle: { fontSize: 14, fontWeight: '700', color: TEXT, marginBottom: 8 },
  infoCardBody: { fontSize: 13, color: TEXT, lineHeight: 21 },
  smallMuted: { fontSize: 12, color: MUTED, marginTop: 6 },

  // 얼굴 사진
  faceGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  faceImg: {
    width: FACE_SIZE, height: FACE_SIZE, borderRadius: 8, backgroundColor: '#eee',
  },

  // 태그
  tagsWrap: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  kwTag: {
    backgroundColor: 'rgba(0,0,0,0.04)', borderRadius: 12,
    paddingHorizontal: 10, paddingVertical: 4,
  },
  kwTagText: { fontSize: 12, color: TEXT },
  locTag: { backgroundColor: 'rgba(122,162,247,0.1)' },
  actTag: { backgroundColor: 'rgba(214,168,95,0.1)' },
});
