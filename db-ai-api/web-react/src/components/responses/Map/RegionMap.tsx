import { useState, useCallback, useMemo } from 'react';
import { GoogleMap, useJsApiLoader, Marker, InfoWindow } from '@react-google-maps/api';
import { MapResponse, MapMarker } from '../../../types/responses';
import { UKRAINE_CENTER, UKRAINE_ZOOM, getRegionByCode } from '../../../constants/regions';

// Map container style
const containerStyle = {
  width: '100%',
  height: '400px',
  borderRadius: '12px',
};

// Map options for clean look
const mapOptions: google.maps.MapOptions = {
  disableDefaultUI: false,
  zoomControl: true,
  mapTypeControl: false,
  streetViewControl: false,
  fullscreenControl: true,
  styles: [
    {
      featureType: 'administrative',
      elementType: 'geometry.stroke',
      stylers: [{ color: '#c9b2a6' }],
    },
    {
      featureType: 'water',
      elementType: 'geometry.fill',
      stylers: [{ color: '#b3d1ff' }],
    },
  ],
};

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
  // Note: _mapType is reserved for future heatmap/choropleth support
  const [selectedMarker, setSelectedMarker] = useState<MapMarker | null>(null);

  // Load Google Maps API
  const { isLoaded, loadError } = useJsApiLoader({
    googleMapsApiKey: import.meta.env.VITE_GOOGLE_MAPS_API_KEY || '',
  });

  // Calculate marker sizes based on values
  const { minValue, maxValue } = useMemo(() => {
    if (!markers.length) return { minValue: 0, maxValue: 1 };
    const values = markers.map((m) => m.value);
    return {
      minValue: Math.min(...values),
      maxValue: Math.max(...values),
    };
  }, [markers]);

  // Get marker scale (20-50px based on value)
  const getMarkerScale = useCallback(
    (value: number) => {
      if (maxValue === minValue) return 35;
      const normalized = (value - minValue) / (maxValue - minValue);
      return 20 + normalized * 30;
    },
    [minValue, maxValue]
  );

  // Format value for display
  const formatValue = useCallback(
    (value: number) => {
      switch (valueFormat) {
        case 'currency':
          return new Intl.NumberFormat('uk-UA', {
            style: 'currency',
            currency: 'UAH',
            maximumFractionDigits: 0,
          }).format(value);
        case 'percent':
          return `${value.toFixed(1)}%`;
        default:
          return new Intl.NumberFormat('uk-UA').format(value);
      }
    },
    [valueFormat]
  );

  // Handle marker click
  const handleMarkerClick = useCallback(
    (marker: MapMarker) => {
      setSelectedMarker(marker);
      if (interactive && onRegionClick) {
        onRegionClick(marker.regionCode);
      }
    },
    [interactive, onRegionClick]
  );

  // Error state
  if (loadError) {
    return (
      <div className="app-card p-6">
        <div className="text-rose-500 flex items-center gap-2">
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <span>Помилка завантаження Google Maps</span>
        </div>
        <p className="text-slate-400 text-sm mt-2">
          Перевірте налаштування API ключа в VITE_GOOGLE_MAPS_API_KEY
        </p>
      </div>
    );
  }

  // Loading state
  if (!isLoaded) {
    return (
      <div className="app-card p-6">
        <div className="flex items-center gap-3 text-slate-400">
          <div className="animate-spin w-5 h-5 border-2 border-sky-400 border-t-transparent rounded-full" />
          <span>Завантаження карти...</span>
        </div>
      </div>
    );
  }

  // No API key warning
  if (!import.meta.env.VITE_GOOGLE_MAPS_API_KEY) {
    return (
      <div className="app-card p-6">
        <div className="text-amber-500 flex items-center gap-2 mb-3">
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <span>Google Maps API ключ не налаштований</span>
        </div>

        {/* Show data as fallback */}
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mt-4">
          {markers.map((marker) => {
            const region = getRegionByCode(marker.regionCode);
            return (
              <div
                key={marker.regionCode}
                className="bg-slate-800/50 rounded-lg p-3 border border-slate-700"
                style={{ borderLeftColor: region?.color, borderLeftWidth: '3px' }}
              >
                <div className="text-lg font-semibold text-white">
                  {formatValue(marker.value)}
                </div>
                <div className="text-sm text-slate-400">
                  {region?.nameUk || marker.regionCode}
                </div>
                <div className="text-xs text-slate-500">{marker.label}</div>
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  return (
    <div className="app-card overflow-hidden">
      {title && (
        <div className="px-4 py-3 border-b border-slate-700/50">
          <h3 className="text-lg font-medium text-white flex items-center gap-2">
            <svg className="w-5 h-5 text-sky-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
            </svg>
            {title}
          </h3>
        </div>
      )}

      <div style={{ height: `${height}px` }}>
        <GoogleMap
          mapContainerStyle={{ ...containerStyle, height: `${height}px` }}
          center={UKRAINE_CENTER}
          zoom={UKRAINE_ZOOM}
          options={mapOptions}
        >
          {markers.map((marker) => {
            const region = getRegionByCode(marker.regionCode);
            if (!region) return null;

            const scale = getMarkerScale(marker.value);

            return (
              <Marker
                key={marker.regionCode}
                position={{ lat: region.lat, lng: region.lng }}
                onClick={() => handleMarkerClick(marker)}
                icon={{
                  path: google.maps.SymbolPath.CIRCLE,
                  scale: scale / 5,
                  fillColor: region.color,
                  fillOpacity: 0.8,
                  strokeColor: '#ffffff',
                  strokeWeight: 2,
                }}
                label={{
                  text: marker.regionCode,
                  color: '#ffffff',
                  fontSize: '11px',
                  fontWeight: 'bold',
                }}
              />
            );
          })}

          {selectedMarker && (
            <InfoWindow
              position={{
                lat: getRegionByCode(selectedMarker.regionCode)?.lat || 0,
                lng: getRegionByCode(selectedMarker.regionCode)?.lng || 0,
              }}
              onCloseClick={() => setSelectedMarker(null)}
            >
              <div className="p-2 min-w-[150px]">
                <h4 className="font-semibold text-gray-900 mb-1">
                  {getRegionByCode(selectedMarker.regionCode)?.nameUk}
                </h4>
                <div className="text-lg font-bold text-sky-600">
                  {formatValue(selectedMarker.value)}
                </div>
                <div className="text-sm text-gray-600">{selectedMarker.label}</div>
                {selectedMarker.secondaryValue !== undefined && (
                  <div className="mt-1 text-sm text-gray-500">
                    {formatValue(selectedMarker.secondaryValue)} {selectedMarker.secondaryLabel}
                  </div>
                )}
                {interactive && (
                  <button
                    className="mt-2 text-xs text-sky-600 hover:text-sky-700 underline"
                    onClick={() => onRegionClick?.(selectedMarker.regionCode)}
                  >
                    Фільтрувати по регіону
                  </button>
                )}
              </div>
            </InfoWindow>
          )}
        </GoogleMap>
      </div>

      {/* Legend */}
      <div className="px-4 py-3 border-t border-slate-700/50 flex flex-wrap gap-3">
        {markers.map((marker) => {
          const region = getRegionByCode(marker.regionCode);
          return (
            <div
              key={marker.regionCode}
              className="flex items-center gap-2 text-sm cursor-pointer hover:bg-slate-700/30 px-2 py-1 rounded transition-colors"
              onClick={() => handleMarkerClick(marker)}
            >
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: region?.color }}
              />
              <span className="text-slate-300">{region?.nameUk}</span>
              <span className="text-slate-500">{formatValue(marker.value)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default RegionMap;
