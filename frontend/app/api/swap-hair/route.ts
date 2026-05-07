const DEFAULT_BACKEND_URL = "http://localhost:8000";

function getBackendSwapUrl(): string {
  const baseUrl = process.env.HAIRSWAP_BACKEND_URL ?? DEFAULT_BACKEND_URL;
  return `${baseUrl.replace(/\/+$/, "")}/api/v1/swap-hair`;
}

function errorResponse(message: string, status: number): Response {
  return Response.json({ detail: message }, { status });
}

export async function POST(request: Request): Promise<Response> {
  let formData: FormData;
  try {
    formData = await request.formData();
  } catch {
    return errorResponse("Request must be multipart/form-data.", 400);
  }

  const baseImage = formData.get("base_image");
  const donorImage = formData.get("donor_image");

  if (!(baseImage instanceof File) || !(donorImage instanceof File)) {
    return errorResponse("Both base_image and donor_image files are required.", 400);
  }

  const outboundForm = new FormData();
  outboundForm.append("base_image", baseImage, baseImage.name);
  outboundForm.append("donor_image", donorImage, donorImage.name);

  const headers = new Headers();
  const apiKey = process.env.HAIRSWAP_API_KEY;

  if (apiKey) {
    headers.set("X-Api-Key", apiKey);
  }

  let backendResponse: Response;

  try {
    backendResponse = await fetch(getBackendSwapUrl(), {
      method: "POST",
      headers,
      body: outboundForm,
      cache: "no-store",
    });
  } catch {
    return errorResponse("Could not reach hair swap backend.", 502);
  }

  const responseHeaders = new Headers();
  const contentType = backendResponse.headers.get("content-type");
  const contentDisposition = backendResponse.headers.get("content-disposition");

  if (contentType) {
    responseHeaders.set("content-type", contentType);
  }
  if (contentDisposition) {
    responseHeaders.set("content-disposition", contentDisposition);
  }

  for (const [headerName, value] of backendResponse.headers.entries()) {
    if (headerName.startsWith("x-pipeline-")) {
      responseHeaders.set(headerName, value);
    }
  }

  const body = await backendResponse.arrayBuffer();
  return new Response(body, {
    status: backendResponse.status,
    headers: responseHeaders,
  });
}
