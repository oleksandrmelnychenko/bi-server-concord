export interface RegionInfo {
  code: string;
  name: string;
  nameUk: string;
  lat: number;
  lng: number;
  color: string;
}

export const UKRAINE_REGIONS: Record<string, RegionInfo> = {
  KI: { code: 'KI', name: 'Kyiv', nameUk: 'Київ', lat: 50.4501, lng: 30.5234, color: '#3B82F6' },
  KD: { code: 'KD', name: 'Kyiv Oblast', nameUk: 'Київська обл.', lat: 50.0536, lng: 30.7667, color: '#0EA5E9' },
  LV: { code: 'LV', name: 'Lviv', nameUk: 'Львів', lat: 49.8397, lng: 24.0297, color: '#8B5CF6' },
  OD: { code: 'OD', name: 'Odesa', nameUk: 'Одеса', lat: 46.4825, lng: 30.7233, color: '#F59E0B' },
  XV: { code: 'XV', name: 'Kharkiv', nameUk: 'Харків', lat: 49.9935, lng: 36.2304, color: '#EF4444' },
  DP: { code: 'DP', name: 'Dnipro', nameUk: 'Дніпро', lat: 48.4647, lng: 35.0462, color: '#06B6D4' },
  XM: { code: 'XM', name: 'Khmelnytskyi', nameUk: 'Хмельницький', lat: 49.423, lng: 26.9871, color: '#10B981' },
  VI: { code: 'VI', name: 'Vinnytsia', nameUk: 'Вінниця', lat: 49.2331, lng: 28.4682, color: '#EC4899' },
  VL: { code: 'VL', name: 'Volyn', nameUk: 'Волинь', lat: 50.7472, lng: 25.3254, color: '#6366F1' },
  DN: { code: 'DN', name: 'Donetsk', nameUk: 'Донецьк', lat: 48.0159, lng: 37.8029, color: '#64748B' },
  GT: { code: 'GT', name: 'Zhytomyr', nameUk: 'Житомир', lat: 50.2547, lng: 28.6587, color: '#14B8A6' },
  ZK: { code: 'ZK', name: 'Zakarpattia', nameUk: 'Закарпаття', lat: 48.6208, lng: 22.2879, color: '#F97316' },
  ZP: { code: 'ZP', name: 'Zaporizhzhia', nameUk: 'Запоріжжя', lat: 47.8388, lng: 35.1396, color: '#84CC16' },
  IF: { code: 'IF', name: 'Ivano-Frankivsk', nameUk: 'Івано-Франківськ', lat: 48.9226, lng: 24.7111, color: '#F43F5E' },
  KR: { code: 'KR', name: 'Kirovohrad', nameUk: 'Кропивницький', lat: 48.5079, lng: 32.2623, color: '#EAB308' },
  LK: { code: 'LK', name: 'Luhansk', nameUk: 'Луганськ', lat: 48.574, lng: 39.3078, color: '#64748B' },
  MI: { code: 'MI', name: 'Mykolaiv', nameUk: 'Миколаїв', lat: 46.975, lng: 31.9946, color: '#D946EF' },
  PL: { code: 'PL', name: 'Poltava', nameUk: 'Полтава', lat: 49.5883, lng: 34.5514, color: '#22C55E' },
  RI: { code: 'RI', name: 'Rivne', nameUk: 'Рівне', lat: 50.6199, lng: 26.2516, color: '#A855F7' },
  SM: { code: 'SM', name: 'Sumy', nameUk: 'Суми', lat: 50.9077, lng: 34.7981, color: '#14B8A6' },
  TE: { code: 'TE', name: 'Ternopil', nameUk: 'Тернопіль', lat: 49.5535, lng: 25.5948, color: '#F97316' },
  XN: { code: 'XN', name: 'Kherson', nameUk: 'Херсон', lat: 46.6354, lng: 32.6169, color: '#84CC16' },
  CK: { code: 'CK', name: 'Cherkasy', nameUk: 'Черкаси', lat: 49.4444, lng: 32.0598, color: '#F43F5E' },
  CE: { code: 'CE', name: 'Chernivtsi', nameUk: 'Чернівці', lat: 48.2921, lng: 25.9358, color: '#6366F1' },
  CN: { code: 'CN', name: 'Chernihiv', nameUk: 'Чернігів', lat: 51.4982, lng: 31.2893, color: '#0EA5E9' },
  PKI: { code: 'PKI', name: 'Kyiv (P)', nameUk: 'Київ (П)', lat: 50.45, lng: 30.52, color: '#3B82F6' },
  PLV: { code: 'PLV', name: 'Lviv (P)', nameUk: 'Львів (П)', lat: 49.84, lng: 24.03, color: '#8B5CF6' },
  POD: { code: 'POD', name: 'Odesa (P)', nameUk: 'Одеса (П)', lat: 46.48, lng: 30.72, color: '#F59E0B' },
  PXV: { code: 'PXV', name: 'Kharkiv (P)', nameUk: 'Харків (П)', lat: 49.99, lng: 36.23, color: '#EF4444' },
  PDP: { code: 'PDP', name: 'Dnipro (P)', nameUk: 'Дніпро (П)', lat: 48.46, lng: 35.05, color: '#06B6D4' },
  PXM: { code: 'PXM', name: 'Khmelnytskyi (P)', nameUk: 'Хмельницький (П)', lat: 49.42, lng: 26.99, color: '#10B981' },
  PVI: { code: 'PVI', name: 'Vinnytsia (P)', nameUk: 'Вінниця (П)', lat: 49.23, lng: 28.47, color: '#EC4899' },
  PVL: { code: 'PVL', name: 'Volyn (P)', nameUk: 'Волинь (П)', lat: 50.75, lng: 25.33, color: '#6366F1' },
  PDN: { code: 'PDN', name: 'Donetsk (P)', nameUk: 'Донецьк (П)', lat: 48.02, lng: 37.8, color: '#64748B' },
  PGT: { code: 'PGT', name: 'Zhytomyr (P)', nameUk: 'Житомир (П)', lat: 50.25, lng: 28.66, color: '#14B8A6' },
  PZK: { code: 'PZK', name: 'Zakarpattia (P)', nameUk: 'Закарпаття (П)', lat: 48.62, lng: 22.29, color: '#F97316' },
  PZP: { code: 'PZP', name: 'Zaporizhzhia (P)', nameUk: 'Запоріжжя (П)', lat: 47.84, lng: 35.14, color: '#84CC16' },
  PIF: { code: 'PIF', name: 'Ivano-Frankivsk (P)', nameUk: 'Івано-Франківськ (П)', lat: 48.92, lng: 24.71, color: '#F43F5E' },
  PKR: { code: 'PKR', name: 'Kirovohrad (P)', nameUk: 'Кропивницький (П)', lat: 48.51, lng: 32.26, color: '#EAB308' },
  PLK: { code: 'PLK', name: 'Luhansk (P)', nameUk: 'Луганськ (П)', lat: 48.57, lng: 39.31, color: '#64748B' },
  PMI: { code: 'PMI', name: 'Mykolaiv (P)', nameUk: 'Миколаїв (П)', lat: 46.97, lng: 32.0, color: '#D946EF' },
  PRI: { code: 'PRI', name: 'Rivne (P)', nameUk: 'Рівне (П)', lat: 50.62, lng: 26.25, color: '#A855F7' },
  PSM: { code: 'PSM', name: 'Sumy (P)', nameUk: 'Суми (П)', lat: 50.91, lng: 34.8, color: '#14B8A6' },
  PTE: { code: 'PTE', name: 'Ternopil (P)', nameUk: 'Тернопіль (П)', lat: 49.55, lng: 25.59, color: '#F97316' },
  PXN: { code: 'PXN', name: 'Kherson (P)', nameUk: 'Херсон (П)', lat: 46.64, lng: 32.62, color: '#84CC16' },
  PCK: { code: 'PCK', name: 'Cherkasy (P)', nameUk: 'Черкаси (П)', lat: 49.44, lng: 32.06, color: '#F43F5E' },
  PCE: { code: 'PCE', name: 'Chernivtsi (P)', nameUk: 'Чернівці (П)', lat: 48.29, lng: 25.94, color: '#6366F1' },
  PCN: { code: 'PCN', name: 'Chernihiv (P)', nameUk: 'Чернігів (П)', lat: 51.5, lng: 31.29, color: '#0EA5E9' },
  PKD: { code: 'PKD', name: 'Kyiv Oblast (P)', nameUk: 'Київська обл. (П)', lat: 50.05, lng: 30.77, color: '#0EA5E9' },
  PMD: { code: 'PMD', name: 'Moldova (P)', nameUk: 'Молдова (П)', lat: 47.01, lng: 28.86, color: '#64748B' },
  PPA: { code: 'PPA', name: 'External (P)', nameUk: 'Зовнішній (П)', lat: 48.38, lng: 31.17, color: '#64748B' },
  PRS: { code: 'PRS', name: 'Russia (P)', nameUk: 'Росія (П)', lat: 55.76, lng: 37.62, color: '#64748B' },
  BL: { code: 'BL', name: 'Belarus', nameUk: 'Білорусь', lat: 53.71, lng: 27.95, color: '#64748B' },
  MD: { code: 'MD', name: 'Moldova', nameUk: 'Молдова', lat: 47.01, lng: 28.86, color: '#64748B' },
  PA: { code: 'PA', name: 'External', nameUk: 'Зовнішній', lat: 48.38, lng: 31.17, color: '#64748B' },
  RS: { code: 'RS', name: 'Russia', nameUk: 'Росія', lat: 55.76, lng: 37.62, color: '#64748B' },
};

export const UKRAINE_CENTER = { lat: 48.3794, lng: 31.1656 };
export const UKRAINE_ZOOM = 6;

export function getRegionByCode(code: string): RegionInfo | undefined {
  return UKRAINE_REGIONS[code.toUpperCase()];
}

export function getAllRegionCodes(): string[] {
  return Object.keys(UKRAINE_REGIONS);
}
