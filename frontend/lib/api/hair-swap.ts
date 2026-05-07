"use client";

const ACCEPTED_IMAGE_TYPES = new Set([
  "image/jpeg",
  "image/jpg",
  "image/png",
  "image/webp",
]);

export class HairSwapError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "HairSwapError";
    this.status = status;
  }
}

type SwapHairParams = {
  baseImage: File;
  donorImage: File;
};

export type StageTelemetry = {
  stage: string;
  implementation: string;
  duration_ms: number;
  models: string[];
};

export type SwapHairResult = {
  imageBlob: Blob;
  stageDetails: StageTelemetry[];
  totalMs: number | null;
};

function validateImage(file: File, fieldName: string): void {
  if (!file) {
    throw new HairSwapError(`${fieldName} is required.`, 400);
  }

  if (!ACCEPTED_IMAGE_TYPES.has(file.type)) {
    throw new HairSwapError(
      `${fieldName} must be JPG, PNG, or WEBP. Received: ${file.type || "unknown"}.`,
      415
    );
  }
}

export async function swapHair({ baseImage, donorImage }: SwapHairParams): Promise<SwapHairResult> {
  validateImage(baseImage, "Base image");
  validateImage(donorImage, "Donor image");

  const formData = new FormData();
  formData.append("base_image", baseImage, baseImage.name);
  formData.append("donor_image", donorImage, donorImage.name);

  const response = await fetch("/api/swap-hair", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const fallback = `Hair swap failed (${response.status}).`;
    const contentType = response.headers.get("content-type") ?? "";

    if (contentType.includes("application/json")) {
      const payload = (await response.json()) as { detail?: string };
      throw new HairSwapError(payload.detail ?? fallback, response.status);
    }

    const text = await response.text();
    throw new HairSwapError(text || fallback, response.status);
  }

  const imageBlob = await response.blob();
  const detailsHeader = response.headers.get("x-pipeline-stage-details");
  const totalHeader = response.headers.get("x-pipeline-ms-total");
  let stageDetails: StageTelemetry[] = [];

  if (detailsHeader) {
    try {
      const parsed = JSON.parse(detailsHeader) as StageTelemetry[];
      if (Array.isArray(parsed)) {
        stageDetails = parsed;
      }
    } catch {
      stageDetails = [];
    }
  }

  const totalMs = totalHeader ? Number(totalHeader) : null;
  return {
    imageBlob,
    stageDetails,
    totalMs: Number.isFinite(totalMs) ? totalMs : null,
  };
}
