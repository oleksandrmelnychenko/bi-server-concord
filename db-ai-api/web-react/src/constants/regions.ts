/**
 * Ukrainian regions configuration for Google Maps integration.
 * Region codes match the database Region.Name values.
 */

export interface RegionInfo {
  code: string;
  name: string;
  nameUk: string;
  lat: number;
  lng: number;
  color: string;
}

export const UKRAINE_REGIONS: Record<string, RegionInfo> = {
  KI: {
    code: 'KI',
    name: 'Kyiv',
    nameUk: 'Київ',
    lat: 50.4501,
    lng: 30.5234,
    color: '#3B82F6', // blue
  },
  XM: {
    code: 'XM',
    name: 'Khmelnytskyi',
    nameUk: 'Хмельницький',
    lat: 49.4230,
    lng: 26.9871,
    color: '#10B981', // emerald
  },
  LV: {
    code: 'LV',
    name: 'Lviv',
    nameUk: 'Львів',
    lat: 49.8397,
    lng: 24.0297,
    color: '#8B5CF6', // violet
  },
  OD: {
    code: 'OD',
    name: 'Odesa',
    nameUk: 'Одеса',
    lat: 46.4825,
    lng: 30.7233,
    color: '#F59E0B', // amber
  },
  XV: {
    code: 'XV',
    name: 'Kharkiv',
    nameUk: 'Харків',
    lat: 49.9935,
    lng: 36.2304,
    color: '#EF4444', // red
  },
  DP: {
    code: 'DP',
    name: 'Dnipro',
    nameUk: 'Дніпро',
    lat: 48.4647,
    lng: 35.0462,
    color: '#06B6D4', // cyan
  },
};

// Ukraine map center (approximately center of the country)
export const UKRAINE_CENTER = {
  lat: 48.3794,
  lng: 31.1656,
};

// Default zoom level to show all of Ukraine
export const UKRAINE_ZOOM = 6;

// Map styling for a clean look
export const MAP_STYLES = [
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
];

/**
 * Get region info by code.
 */
export function getRegionByCode(code: string): RegionInfo | undefined {
  return UKRAINE_REGIONS[code.toUpperCase()];
}

/**
 * Get all region codes.
 */
export function getAllRegionCodes(): string[] {
  return Object.keys(UKRAINE_REGIONS);
}

/**
 * Map Ukrainian region name to code.
 */
export function getRegionCodeByName(name: string): string | undefined {
  const nameLower = name.toLowerCase();

  const nameToCode: Record<string, string> = {
    'київ': 'KI',
    'kyiv': 'KI',
    'хмельницький': 'XM',
    'khmelnytskyi': 'XM',
    'львів': 'LV',
    'lviv': 'LV',
    'одеса': 'OD',
    'odesa': 'OD',
    'харків': 'XV',
    'kharkiv': 'XV',
    'дніпро': 'DP',
    'dnipro': 'DP',
  };

  return nameToCode[nameLower];
}
