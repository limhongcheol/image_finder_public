import { useEffect, useState } from 'react';
import { View, Text, ActivityIndicator, StyleSheet } from 'react-native';
import { WebView } from 'react-native-webview';
import { getLifestyleEvents, mediaUrl } from '../api';
import { useOnline } from '../useOnline';
import OfflineNotice from '../components/OfflineNotice';
import { ACCENT, BG, TEXT, MUTED, PADDING } from '../theme';

// leaflet 지도 HTML (web_app.py 와 동일 방식). markers 는 RN 에서 주입.
// 사진이 많을 때(수천~수만) 개별 마커를 그대로 추가하면 WebView 가 멈추므로
// Leaflet.markercluster 로 클러스터링 + 청크 로딩하여 대량 마커도 부드럽게 처리한다.
function buildHtml(markers) {
  return `<!DOCTYPE html><html><head>
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>
<style>html,body,#map{height:100%;margin:0}
#loading{position:absolute;top:8px;left:50%;transform:translateX(-50%);z-index:1000;
  background:rgba(0,0,0,.65);color:#fff;padding:6px 12px;border-radius:14px;font:13px sans-serif}</style></head>
<body><div id="map"></div><div id="loading">지도 준비 중…</div><script>
var markers = ${JSON.stringify(markers)};
// preferCanvas: 벡터 렌더를 캔버스로 → 대량 요소에서 성능 향상
var map = L.map('map', { preferCanvas: true });
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'OSM'}).addTo(map);

// 클러스터 그룹: 화면 밖 마커 제거 + 청크 단위 추가로 메인 스레드 블로킹 방지
var cluster = L.markerClusterGroup({
  chunkedLoading: true,
  chunkInterval: 100,
  chunkDelay: 30,
  removeOutsideVisibleBounds: true,
  spiderfyOnMaxZoom: true,
  disableClusteringAtZoom: 18,
  maxClusterRadius: 60
});

var bounds = L.latLngBounds([]);
var batch = [];
for (var i = 0; i < markers.length; i++) {
  var m = markers[i];
  if (m.lat == null || m.lon == null) continue;
  var mk = L.marker([m.lat, m.lon]);
  // 팝업 내용은 클릭 시에만 생성(1만개 팝업 HTML 선생성 방지)
  (function(marker, info){
    marker.on('click', function(){
      if (!marker._built) {
        var html = (info.thumb ? '<img src="'+info.thumb+'" style="width:120px;border-radius:6px"/><br>' : '') + (info.name || '');
        marker.bindPopup(html);
        marker._built = true;
      }
      marker.openPopup();
    });
  })(mk, m);
  batch.push(mk);
  bounds.extend([m.lat, m.lon]);
}

cluster.addLayers(batch);          // 청크 로딩으로 한 번에 대량 추가
map.addLayer(cluster);

if (bounds.isValid()) map.fitBounds(bounds, { padding:[40,40] });
else map.setView([37.5,127], 11);

var el = document.getElementById('loading');
if (el) el.parentNode.removeChild(el);
</script></body></html>`;
}

export default function MapScreen() {
  const [loading, setLoading] = useState(true);
  const [html, setHtml] = useState(null);
  const [count, setCount] = useState(0);
  const online = useOnline();   // 지도 타일/leaflet 은 인터넷 필요

  useEffect(() => {
    (async () => {
      const data = await getLifestyleEvents();
      const geo = (data?.events || [])
        .filter(e => e.lat != null && e.lon != null)
        .map(e => ({ lat: e.lat, lon: e.lon, name: e.filename || '', thumb: mediaUrl(e.thumbnail_path) }));
      setCount(geo.length);
      setHtml(buildHtml(geo));
      setLoading(false);
    })();
  }, []);

  return (
    <View style={styles.wrap}>
      <View style={styles.head}>
        <Text style={styles.title}>지도</Text>
        <Text style={styles.sub}>위치 정보가 있는 사진 {count}장</Text>
      </View>
      {online === false ? (
        <OfflineNotice
          title="오프라인 상태예요"
          message="지도는 인터넷(지도 타일)이 필요해요. 연결되면 위치 사진을 지도에서 볼 수 있어요."
        />
      ) : loading ? (
        <View style={styles.center}><ActivityIndicator color={ACCENT} /></View>
      ) : (
        <WebView
          originWhitelist={['*']}
          source={{ html }}
          style={styles.web}
          javaScriptEnabled
          domStorageEnabled
          mixedContentMode="always"
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { flex: 1, backgroundColor: BG },
  head: { paddingHorizontal: PADDING, paddingTop: 8, paddingBottom: 8 },
  title: { fontSize: 24, fontWeight: '700', color: TEXT },
  sub: { fontSize: 13, color: MUTED, marginTop: 4 },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  web: { flex: 1, backgroundColor: '#eee' },
});
