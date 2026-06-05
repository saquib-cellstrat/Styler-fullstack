"use client";

import { useEffect, useMemo, useState } from "react";
import Image from "next/image";
import { AlertCircle, Download, ImagePlus, RefreshCw, Sparkles, Upload, X } from "lucide-react";
import { Section } from "@/components/layout/section";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { type StageTelemetry, swapHair } from "@/lib/api/hair-swap";

type UploadField = "base" | "donor";

type UploadState = {
  file: File | null;
  previewUrl: string | null;
};

const ACCEPTED_TYPES = "image/jpeg,image/jpg,image/png,image/webp";
const EMPTY_STAGE_DETAILS: StageTelemetry[] = [];
const DEFAULT_BASE_IMAGE_PATH = "/bald-image.png";
const DEFAULT_BASE_IMAGE_FILE = "bald-image.png";

function revokePreviewUrl(url: string | null) {
  if (url?.startsWith("blob:")) {
    URL.revokeObjectURL(url);
  }
}

async function fetchDefaultBaseFile(): Promise<File> {
  const response = await fetch(DEFAULT_BASE_IMAGE_PATH);
  if (!response.ok) {
    throw new Error(`Could not load default base image (${response.status}).`);
  }
  const blob = await response.blob();
  const type = blob.type || "image/png";
  return new File([blob], DEFAULT_BASE_IMAGE_FILE, { type });
}

function formatFileMeta(file: File | null): string {
  if (!file) {
    return "No file selected";
  }
  const sizeMb = (file.size / 1024 / 1024).toFixed(2);
  return `${file.name} (${sizeMb} MB)`;
}

function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.rel = "noopener";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function canvasToPngBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) {
        resolve(blob);
        return;
      }
      reject(new Error("Could not build collage image."));
    }, "image/png");
  });
}

/** One row: base | donor | result, each fitted inside the same cell size. */
async function buildHairSwapCollagePng(
  baseFile: File,
  donorFile: File,
  resultBlob: Blob,
): Promise<Blob> {
  let baseBm: ImageBitmap;
  let donorBm: ImageBitmap;
  let resultBm: ImageBitmap;
  try {
    [baseBm, donorBm, resultBm] = await Promise.all([
      createImageBitmap(baseFile),
      createImageBitmap(donorFile),
      createImageBitmap(resultBlob),
    ]);
  } catch {
    throw new Error("Could not decode one of the images for the collage.");
  }

  try {
    const gap = 24;
    const pad = 24;
    const colW = 720;
    const rowH = 900;

    const canvas = document.createElement("canvas");
    canvas.width = pad * 2 + colW * 3 + gap * 2;
    canvas.height = pad * 2 + rowH;

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      throw new Error("Canvas is not available.");
    }

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const panels: ImageBitmap[] = [baseBm, donorBm, resultBm];
    for (let i = 0; i < panels.length; i += 1) {
      const img = panels[i];
      const x0 = pad + i * (colW + gap);
      const scale = Math.min(colW / img.width, rowH / img.height);
      const dw = img.width * scale;
      const dh = img.height * scale;
      const dx = x0 + (colW - dw) / 2;
      const dy = pad + (rowH - dh) / 2;
      ctx.drawImage(img, dx, dy, dw, dh);
    }

    return await canvasToPngBlob(canvas);
  } finally {
    baseBm.close();
    donorBm.close();
    resultBm.close();
  }
}

export default function SwapPage() {
  const [base, setBase] = useState<UploadState>({ file: null, previewUrl: null });
  const [defaultBaseError, setDefaultBaseError] = useState<string | null>(null);
  const [donor, setDonor] = useState<UploadState>({ file: null, previewUrl: null });
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [stageDetails, setStageDetails] = useState<StageTelemetry[]>(EMPTY_STAGE_DETAILS);
  const [totalMs, setTotalMs] = useState<number | null>(null);
  const [viewerImage, setViewerImage] = useState<{ src: string; alt: string } | null>(null);
  const [isDownloadingCollage, setIsDownloadingCollage] = useState(false);

  const canSubmit = useMemo(() => Boolean(base.file && donor.file && !isSubmitting), [
    base.file,
    donor.file,
    isSubmitting,
  ]);

  const canDownloadCollage = useMemo(
    () => Boolean(base.file && donor.file && resultUrl && !isDownloadingCollage && !isSubmitting),
    [base.file, donor.file, resultUrl, isDownloadingCollage, isSubmitting],
  );

  useEffect(() => {
    let cancelled = false;
    fetchDefaultBaseFile()
      .then((file) => {
        if (cancelled) return;
        setBase({
          file,
          previewUrl: DEFAULT_BASE_IMAGE_PATH,
        });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        const message = err instanceof Error ? err.message : "Could not load default base image.";
        setDefaultBaseError(message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    return () => {
      revokePreviewUrl(base.previewUrl);
      revokePreviewUrl(donor.previewUrl);
      revokePreviewUrl(resultUrl);
    };
  }, [base.previewUrl, donor.previewUrl, resultUrl]);

  function handleFileChange(field: UploadField, file: File | null) {
    setErrorMessage(null);
    if (resultUrl) {
      revokePreviewUrl(resultUrl);
      setResultUrl(null);
    }
    setStageDetails(EMPTY_STAGE_DETAILS);
    setTotalMs(null);

    const nextState: UploadState = {
      file,
      previewUrl: file ? URL.createObjectURL(file) : null,
    };

    if (field === "base") {
      revokePreviewUrl(base.previewUrl);
      if (!file) {
        void fetchDefaultBaseFile()
          .then((f) => {
            setDefaultBaseError(null);
            setBase({ file: f, previewUrl: DEFAULT_BASE_IMAGE_PATH });
          })
          .catch((err: unknown) => {
            const message = err instanceof Error ? err.message : "Could not load default base image.";
            setDefaultBaseError(message);
            setBase({ file: null, previewUrl: null });
          });
        return;
      }
      setBase({ file, previewUrl: URL.createObjectURL(file) });
      return;
    }

    revokePreviewUrl(donor.previewUrl);
    setDonor(nextState);
  }

  async function onSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setErrorMessage(null);

    if (!base.file || !donor.file) {
      setErrorMessage("Please choose a donor image. The default base should load automatically.");
      return;
    }

    setIsSubmitting(true);

    try {
      const swapResult = await swapHair({ baseImage: base.file, donorImage: donor.file });
      if (resultUrl) {
        revokePreviewUrl(resultUrl);
      }
      setResultUrl(URL.createObjectURL(swapResult.imageBlob));
      setStageDetails(swapResult.stageDetails);
      setTotalMs(swapResult.totalMs);
    } catch (error) {
      if (error instanceof Error) {
        setErrorMessage(error.message);
      } else {
        setErrorMessage("Hair swap failed. Please try again.");
      }
    } finally {
      setIsSubmitting(false);
    }
  }

  async function resetAll() {
    setErrorMessage(null);
    setDefaultBaseError(null);
    revokePreviewUrl(base.previewUrl);
    revokePreviewUrl(donor.previewUrl);
    revokePreviewUrl(resultUrl);
    setDonor({ file: null, previewUrl: null });
    setResultUrl(null);
    setStageDetails(EMPTY_STAGE_DETAILS);
    setTotalMs(null);
    setBase({ file: null, previewUrl: null });
    try {
      const file = await fetchDefaultBaseFile();
      setBase({ file, previewUrl: DEFAULT_BASE_IMAGE_PATH });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Could not load default base image.";
      setDefaultBaseError(message);
    }
  }

  async function downloadCollage() {
    if (!base.file || !donor.file || !resultUrl) {
      return;
    }
    setIsDownloadingCollage(true);
    try {
      const resultResponse = await fetch(resultUrl);
      if (!resultResponse.ok) {
        throw new Error("Could not read the swap result for the collage.");
      }
      const resultBlob = await resultResponse.blob();
      const collageBlob = await buildHairSwapCollagePng(base.file, donor.file, resultBlob);
      downloadBlob(collageBlob, "hair-swap-collage.png");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Could not create collage.";
      setErrorMessage(message);
    } finally {
      setIsDownloadingCollage(false);
    }
  }

  return (
    <div className="bg-background text-foreground">
      <Section className="py-12 md:py-16" containerClassName="space-y-8">
        <header className="space-y-3">
          <p className="inline-flex items-center gap-2 text-sm text-muted-foreground">
            <Sparkles className="h-4 w-4 text-accent" />
            Hair Swap Studio
          </p>
          <h1 className="font-display text-4xl leading-tight md:text-5xl">Swap Hair</h1>
          <p className="max-w-3xl text-muted-foreground">
            The default base is the studio bald reference. Upload a donor photo to run a swap, or
            replace the base with your own image.
          </p>
        </header>

        {defaultBaseError ? (
          <div className="flex items-start gap-3 rounded-xl border border-amber-300 bg-amber-50 p-4 text-amber-900 dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-100">
            <AlertCircle className="mt-0.5 h-5 w-5 shrink-0" />
            <p className="text-sm">{defaultBaseError}</p>
          </div>
        ) : null}

        <form onSubmit={onSubmit} className="grid gap-6 lg:grid-cols-2">
          <div className="space-y-4 rounded-xl border border-border bg-card p-5 shadow-sm">
            <label htmlFor="base-image" className="text-sm font-medium">
              Base image
            </label>
            <Input
              id="base-image"
              type="file"
              accept={ACCEPTED_TYPES}
              onChange={(event) => handleFileChange("base", event.target.files?.[0] ?? null)}
            />
            <p className="text-sm text-muted-foreground">{formatFileMeta(base.file)}</p>
            {base.previewUrl ? (
              <button
                type="button"
                className="block w-full text-left"
                onClick={() => setViewerImage({ src: base.previewUrl as string, alt: "Base preview" })}
              >
                <ImagePreview src={base.previewUrl as string} alt="Base preview" fit="cover" />
              </button>
            ) : (
              <div className="aspect-[4/3] overflow-hidden rounded-lg border border-border bg-muted/40">
                <div className="flex h-full items-center justify-center gap-2 text-sm text-muted-foreground">
                  <ImagePlus className="h-4 w-4" />
                  Base preview
                </div>
              </div>
            )}
          </div>

          <div className="space-y-4 rounded-xl border border-border bg-card p-5 shadow-sm">
            <label htmlFor="donor-image" className="text-sm font-medium">
              Donor image
            </label>
            <Input
              id="donor-image"
              type="file"
              accept={ACCEPTED_TYPES}
              onChange={(event) => handleFileChange("donor", event.target.files?.[0] ?? null)}
            />
            <p className="text-sm text-muted-foreground">{formatFileMeta(donor.file)}</p>
            {donor.previewUrl ? (
              <button
                type="button"
                className="block w-full text-left"
                onClick={() =>
                  setViewerImage({ src: donor.previewUrl as string, alt: "Donor preview" })
                }
              >
                <ImagePreview src={donor.previewUrl as string} alt="Donor preview" fit="cover" />
              </button>
            ) : (
              <div className="aspect-[4/3] overflow-hidden rounded-lg border border-border bg-muted/40">
                <div className="flex h-full items-center justify-center gap-2 text-sm text-muted-foreground">
                  <ImagePlus className="h-4 w-4" />
                  Donor preview
                </div>
              </div>
            )}
          </div>

          <div className="lg:col-span-2 flex flex-wrap items-center gap-3">
            <Button type="submit" disabled={!canSubmit}>
              <Upload className="h-4 w-4" />
              {isSubmitting ? "Processing..." : "Generate Hair Swap"}
            </Button>
            <Button type="button" variant="secondary" onClick={resetAll} disabled={isSubmitting}>
              <RefreshCw className="h-4 w-4" />
              Reset
            </Button>
            <Button
              type="button"
              variant="outline"
              onClick={() => void downloadCollage()}
              disabled={!canDownloadCollage}
              title="Save one image: base, donor, and result side by side"
            >
              <Download className="h-4 w-4" />
              {isDownloadingCollage ? "Building…" : "Download collage"}
            </Button>
          </div>
        </form>

        {errorMessage ? (
          <div className="flex items-start gap-3 rounded-xl border border-red-300 bg-red-50 p-4 text-red-700 dark:border-red-800 dark:bg-red-950/40 dark:text-red-200">
            <AlertCircle className="mt-0.5 h-5 w-5 shrink-0" />
            <p className="text-sm">{errorMessage}</p>
          </div>
        ) : null}

        <section className="space-y-3">
          <h2 className="text-xl font-semibold">Result</h2>
          <div className="aspect-[16/10] overflow-hidden rounded-xl border border-border bg-muted/30">
            {resultUrl ? (
              <button
                type="button"
                className="block h-full w-full text-left"
                onClick={() => setViewerImage({ src: resultUrl, alt: "Hair swap result" })}
              >
                <ImagePreview src={resultUrl} alt="Hair swap result" fit="contain" />
              </button>
            ) : (
              <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
                Final image appears here after processing.
              </div>
            )}
          </div>
        </section>

        <section className="space-y-3">
          <h2 className="text-xl font-semibold">Stage Performance</h2>
          <div className="rounded-xl border border-border bg-card p-4">
            {stageDetails.length ? (
              <div className="space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-border bg-muted/30 px-3 py-2">
                  <span className="text-sm text-muted-foreground">Pipeline summary</span>
                  <span className="text-xs text-muted-foreground">
                    Detailed per-stage execution metrics
                  </span>
                </div>
                <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
                  <span>Total pipeline time:</span>
                  <span className="font-semibold text-foreground">
                    {totalMs !== null ? `${totalMs.toFixed(3)} ms` : "N/A"}
                  </span>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[640px] text-left text-sm">
                    <thead className="text-muted-foreground">
                      <tr>
                        <th className="pb-2 pr-4 font-medium">Stage</th>
                        <th className="pb-2 pr-4 font-medium">Implementation</th>
                        <th className="pb-2 pr-4 font-medium">Time (ms)</th>
                        <th className="pb-2 font-medium">Models</th>
                      </tr>
                    </thead>
                    <tbody>
                      {stageDetails.map((item) => (
                        <tr key={item.stage} className="border-t border-border">
                          <td className="py-2 pr-4 font-medium capitalize">{item.stage}</td>
                          <td className="py-2 pr-4">{item.implementation || "N/A"}</td>
                          <td className="py-2 pr-4">{item.duration_ms.toFixed(3)}</td>
                          <td className="py-2">{item.models.join(", ") || "N/A"}</td>
                        </tr>
                      ))}
                      <tr className="border-t-2 border-border bg-muted/20">
                        <td className="py-2 pr-4 font-semibold">Total</td>
                        <td className="py-2 pr-4 text-muted-foreground">All stages</td>
                        <td className="py-2 pr-4 font-semibold">
                          {totalMs !== null ? totalMs.toFixed(3) : "N/A"}
                        </td>
                        <td className="py-2 text-muted-foreground">Aggregate</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">
                Run a swap to view detailed stage timings and model usage.
              </p>
            )}
          </div>
        </section>
      </Section>
      <ImageViewerModal
        image={viewerImage}
        onClose={() => setViewerImage(null)}
      />
    </div>
  );
}

function ImagePreview({
  src,
  alt,
  fit,
}: {
  src: string;
  alt: string;
  fit: "cover" | "contain";
}) {
  return (
    <div className="relative aspect-[4/3] overflow-hidden rounded-lg border border-border bg-muted/40">
      <Image
        src={src}
        alt={alt}
        fill
        unoptimized
        className={fit === "cover" ? "object-cover" : "object-contain"}
        sizes="(max-width: 1024px) 100vw, 50vw"
      />
      <div className="absolute bottom-2 right-2 rounded-md bg-background/90 px-2 py-1 text-xs text-muted-foreground">
        Click to open viewer
      </div>
    </div>
  );
}

function ImageViewerModal({
  image,
  onClose,
}: {
  image: { src: string; alt: string } | null;
  onClose: () => void;
}) {
  const [zoom, setZoom] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const [start, setStart] = useState({ x: 0, y: 0 });

  if (!image) {
    return null;
  }

  return (
    <div className="fixed inset-0 z-50 bg-black/80 p-4 md:p-8">
      <div className="mx-auto flex h-full w-full max-w-6xl flex-col gap-3">
        <div className="flex items-center justify-between rounded-lg bg-card px-4 py-3">
          <p className="text-sm text-muted-foreground">{image.alt}</p>
          <button
            type="button"
            onClick={onClose}
            className="inline-flex h-10 w-10 items-center justify-center rounded-lg border border-border bg-background text-foreground"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div
          className="relative flex-1 cursor-grab overflow-hidden rounded-lg border border-border bg-black active:cursor-grabbing"
          onMouseDown={(event) => {
            setDragging(true);
            setStart({ x: event.clientX - position.x, y: event.clientY - position.y });
          }}
          onMouseMove={(event) => {
            if (!dragging) return;
            setPosition({ x: event.clientX - start.x, y: event.clientY - start.y });
          }}
          onMouseUp={() => setDragging(false)}
          onMouseLeave={() => setDragging(false)}
        >
          <div
            className="absolute inset-0"
            style={{
              transform: `translate(${position.x}px, ${position.y}px) scale(${zoom})`,
              transformOrigin: "center center",
              transition: dragging ? "none" : "transform 120ms ease-out",
            }}
          >
            <Image
              src={image.src}
              alt={image.alt}
              fill
              unoptimized
              className="object-contain"
              sizes="100vw"
            />
          </div>
        </div>

        <div className="flex items-center gap-3 rounded-lg bg-card px-4 py-3">
          <label htmlFor="viewer-zoom" className="text-sm text-muted-foreground">
            Zoom
          </label>
          <input
            id="viewer-zoom"
            type="range"
            min={1}
            max={6}
            step={0.1}
            value={zoom}
            onChange={(event) => setZoom(Number(event.target.value))}
            className="w-full accent-accent"
          />
          <span className="w-16 text-right text-sm font-medium">{zoom.toFixed(1)}x</span>
        </div>
      </div>
    </div>
  );
}
