"use client";

import { useEffect, useMemo, useState } from "react";
import Image from "next/image";
import { AlertCircle, ImagePlus, RefreshCw, Sparkles, Upload, X } from "lucide-react";
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

function formatFileMeta(file: File | null): string {
  if (!file) {
    return "No file selected";
  }
  const sizeMb = (file.size / 1024 / 1024).toFixed(2);
  return `${file.name} (${sizeMb} MB)`;
}

export default function SwapPage() {
  const [base, setBase] = useState<UploadState>({ file: null, previewUrl: null });
  const [donor, setDonor] = useState<UploadState>({ file: null, previewUrl: null });
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [stageDetails, setStageDetails] = useState<StageTelemetry[]>(EMPTY_STAGE_DETAILS);
  const [totalMs, setTotalMs] = useState<number | null>(null);
  const [viewerImage, setViewerImage] = useState<{ src: string; alt: string } | null>(null);

  const canSubmit = useMemo(() => Boolean(base.file && donor.file && !isSubmitting), [
    base.file,
    donor.file,
    isSubmitting,
  ]);

  useEffect(() => {
    return () => {
      if (base.previewUrl) {
        URL.revokeObjectURL(base.previewUrl);
      }
      if (donor.previewUrl) {
        URL.revokeObjectURL(donor.previewUrl);
      }
      if (resultUrl) {
        URL.revokeObjectURL(resultUrl);
      }
    };
  }, [base.previewUrl, donor.previewUrl, resultUrl]);

  function handleFileChange(field: UploadField, file: File | null) {
    setErrorMessage(null);
    if (resultUrl) {
      URL.revokeObjectURL(resultUrl);
      setResultUrl(null);
    }
    setStageDetails(EMPTY_STAGE_DETAILS);
    setTotalMs(null);

    const nextState: UploadState = {
      file,
      previewUrl: file ? URL.createObjectURL(file) : null,
    };

    if (field === "base") {
      if (base.previewUrl) {
        URL.revokeObjectURL(base.previewUrl);
      }
      setBase(nextState);
      return;
    }

    if (donor.previewUrl) {
      URL.revokeObjectURL(donor.previewUrl);
    }
    setDonor(nextState);
  }

  async function onSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setErrorMessage(null);

    if (!base.file || !donor.file) {
      setErrorMessage("Please choose both a base image and a donor image.");
      return;
    }

    setIsSubmitting(true);

    try {
      const swapResult = await swapHair({ baseImage: base.file, donorImage: donor.file });
      if (resultUrl) {
        URL.revokeObjectURL(resultUrl);
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

  function resetAll() {
    setErrorMessage(null);
    if (base.previewUrl) {
      URL.revokeObjectURL(base.previewUrl);
    }
    if (donor.previewUrl) {
      URL.revokeObjectURL(donor.previewUrl);
    }
    if (resultUrl) {
      URL.revokeObjectURL(resultUrl);
    }
    setBase({ file: null, previewUrl: null });
    setDonor({ file: null, previewUrl: null });
    setResultUrl(null);
    setStageDetails(EMPTY_STAGE_DETAILS);
    setTotalMs(null);
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
            Upload a base photo and a donor photo. The model returns a single PNG with the donor
            hair blended onto the base image.
          </p>
        </header>

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
                onClick={() => setViewerImage({ src: base.previewUrl, alt: "Base preview" })}
              >
                <ImagePreview src={base.previewUrl} alt="Base preview" fit="cover" />
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
                onClick={() => setViewerImage({ src: donor.previewUrl, alt: "Donor preview" })}
              >
                <ImagePreview src={donor.previewUrl} alt="Donor preview" fit="cover" />
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
