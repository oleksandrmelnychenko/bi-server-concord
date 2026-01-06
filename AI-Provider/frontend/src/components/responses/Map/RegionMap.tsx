import { useState, useCallback, useMemo, useEffect } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { MapResponse, MapMarker } from '../../../types/responses';
import { UKRAINE_CENTER, UKRAINE_ZOOM, getRegionByCode } from '../../../constants/regions';

import L from 'leaflet';
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

function SetViewOnChange({ center, zoom }: { center: [number, number]; zoom: number }) {
  const map = useMap();
  useEffect(() => {
    map.setView(center, zoom);
  }, [center, zoom, map]);
  return null;
}

interface RegionMapProps extends Omit<MapResponse, 'type'> {
  onRegionClick?: (regionCode: string) => void;
}

export function RegionMap({
  title,
  markers,
  mapType: _mapType = 'markers',
  valueFormat = 'number',
  interactive = true,
  height = 400,
  onRegionClick,
}: RegionMapProps) {
  const [_selectedMarker, setSelectedMarker] = useState<MapMarker | null>(null);

  const totalValue = useMemo(() => {
    if (!markers.length) return 1;
    return markers.reduce((sum, m) => sum + m.value, 0);
  }, [markers]);

  const getMarkerRadius = useCallback(
    (value: number) => {
      if (totalValue === 0) return 15;
      const percent = (value / totalValue) * 100;
      return 8 + Math.sqrt(percent / 100) * 37;
    },
    [totalValue]
  );

  const getPercentage = useCallback(
    (value: number) => {
      if (totalValue === 0) return 0;
      return (value / totalValue) * 100;
    },
    [totalValue]
  );

  const formatValue = useCallback(
    (value: number) => {
      switch (valueFormat) {
        case 'currency':
          return new Intl.NumberFormat('uk-UA', { style: 'currency', currency: 'UAH', maximumFractionDigits: 0 }).format(value);
        case 'percent':
          return value.toFixed(1) + '%';
        default:
          return new Intl.NumberFormat('uk-UA').format(value);
      }
    },
    [valueFormat]
  );

  const handleMarkerClick = useCallback(
    (marker: MapMarker) => {
      setSelectedMarker(marker);
      if (interactive && onRegionClick) onRegionClick(marker.regionCode);
    },
    [interactive, onRegionClick]
  );

  if (!markers.length) {
    return (<div className="app-card p-6"><div className="text-slate-400">Немає даних для відображення на карті</div></div>);
  }

  return (
    <div className="app-card overflow-hidden">
      {title && (
        <div className="px-4 py-3 border-b border-slate-700/50">
          <h3 className="text-lg font-medium text-white">{title}</h3>
        </div>
      )}
      <div style={{ height: height + 'px' }}>
        <MapContainer center={[UKRAINE_CENTER.lat, UKRAINE_CENTER.lng]} zoom={UKRAINE_ZOOM} style={{ height: '100%', width: '100%' }} scrollWheelZoom={true}>
          <SetViewOnChange center={[UKRAINE_CENTER.lat, UKRAINE_CENTER.lng]} zoom={UKRAINE_ZOOM} />
          <TileLayer attribution='OpenStreetMap' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
          {markers.map((marker) => {
            const region = getRegionByCode(marker.regionCode);
            if (!region) return null;
            const radius = getMarkerRadius(marker.value);
            const percent = getPercentage(marker.value);
            return (
              <CircleMarker key={marker.regionCode} center={[region.lat, region.lng]} radius={radius}
                pathOptions={{ fillColor: region.color, fillOpacity: 0.7, color: '#fff', weight: 2 }}
                eventHandlers={{ click: () => handleMarkerClick(marker) }}>
                <Popup>
                  <div className="min-w-[150px]">
                    <h4 className="font-semibold text-gray-900 mb-1">{region.nameUk}</h4>
                    <div className="text-lg font-bold text-sky-600">{formatValue(marker.value)}</div>
                    <div className="text-sm text-emerald-600 font-medium">{percent.toFixed(1)}%</div>
                    <div className="text-sm text-gray-600">{marker.label}</div>
                  </div>
                </Popup>
              </CircleMarker>
            );
          })}
        </MapContainer>
      </div>
      <div className="px-4 py-3 border-t border-slate-700/50 flex flex-wrap gap-3">
        {markers.map((marker) => {
          const region = getRegionByCode(marker.regionCode);
          const percent = getPercentage(marker.value);
          return (
            <div key={marker.regionCode} className="flex items-center gap-2 text-sm cursor-pointer hover:bg-slate-700/30 px-2 py-1 rounded" onClick={() => handleMarkerClick(marker)}>
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: region?.color }} />
              <span className="text-slate-300">{region?.nameUk}</span>
              <span className="text-slate-500">{formatValue(marker.value)}</span>
              <span className="text-emerald-400">({percent.toFixed(1)}%)</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default RegionMap;
