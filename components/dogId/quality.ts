import { PhotoFile } from 'react-native-vision-camera';

export type QualityResult = { ok: true } | { ok: false; reasons: string[] };

export function computeLaplacianVariance(photo: PhotoFile): number {
  const meta = (photo as any)?.metadata ?? {};
  const sharpness = typeof meta.Sharpness === 'number' ? meta.Sharpness : 0;
  const exposure = typeof meta.ExposureTime === 'number' ? meta.ExposureTime : meta.ShutterSpeedValue ?? 0;
  const iso = typeof meta.ISO === 'number' ? meta.ISO : meta.PhotographicSensitivity ?? 0;
  const sizeBoost = Math.log10(Math.max(photo.width * photo.height, 1));

  let score = sharpness + sizeBoost;
  if (exposure > 0.05) score -= 2; // long shutter likely blurrier
  if (iso > 1600) score -= 1.5; // high ISO usually softer/noisier
  return Math.max(score, 0);
}

export function evaluateQuality(photo: PhotoFile): QualityResult {
  const reasons: string[] = [];
  const minWidth = 1200;
  const minHeight = 1200;

  if (photo.width < minWidth || photo.height < minHeight) {
    reasons.push(`Low resolution: got ${photo.width}x${photo.height}, need at least ${minWidth}x${minHeight}.`);
  }

  const brightness = (photo.metadata as any)?.BrightnessValue;
  if (typeof brightness === 'number') {
    if (brightness < 0.5) {
      reasons.push('Too dark: increase light or move to brighter area.');
    } else if (brightness > 8) {
      reasons.push('Too bright: reduce glare or move out of direct sun.');
    }
  }

  const exposureTime = (photo.metadata as any)?.ExposureTime ?? (photo.metadata as any)?.ShutterSpeedValue;
  if (typeof exposureTime === 'number' && exposureTime > 0.033) {
    reasons.push('Possible blur: shutter slower than 1/30s, hold steadier.');
  }

  const iso = (photo.metadata as any)?.ISO ?? (photo.metadata as any)?.PhotographicSensitivity;
  if (typeof iso === 'number' && iso > 1600) {
    reasons.push('High ISO: image may be noisy/soft. Add light or hold steadier.');
  }

  return reasons.length ? { ok: false, reasons } : { ok: true };
}

export function pickBestSharpest(photos: PhotoFile[]): PhotoFile {
  if (photos.length === 0) {
    throw new Error('No photos provided');
  }
  const scored = photos.map(photo => ({ photo, score: computeLaplacianVariance(photo) }));
  return scored.reduce((top, current) => (current.score > top.score ? current : top), scored[0]).photo;
}
