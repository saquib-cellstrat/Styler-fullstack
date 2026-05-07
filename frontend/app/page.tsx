"use client";

import Link from "next/link";
import { ArrowRight, CheckCircle2, Sparkles, WandSparkles } from "lucide-react";
import { Section } from "@/components/layout/section";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

export default function Home() {
  return (
    <div className="bg-background text-foreground">
      <Section className="surface-glow" containerClassName="space-y-10">
        <div className="space-y-5">
          <Badge pulse>AI Hair Transformation</Badge>
          <h1 className="max-w-4xl font-display text-5xl leading-[1.05] md:text-7xl">
            HairSwap Studio for
            <span className="gradient-text"> fast visual try-ons</span>
          </h1>
          <p className="max-w-3xl text-lg leading-8 text-muted-foreground">
            Upload a base portrait and a donor hairstyle, then generate a realistic blended result
            in seconds. Built for rapid look previews and client-ready experimentation.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <Link href="/swap">
            <Button>
              <WandSparkles className="h-4 w-4" />
              Open Hair Swap Tool
              <ArrowRight className="h-4 w-4" />
            </Button>
          </Link>
          <span className="text-sm text-muted-foreground">
            Works with JPG, PNG, and WEBP images
          </span>
        </div>

        <div className="grid gap-4 md:grid-cols-3">
          <div className="rounded-xl border border-border bg-card p-5">
            <p className="text-sm text-muted-foreground">Pipeline</p>
            <p className="mt-2 text-2xl font-semibold">Alignment + Blend</p>
          </div>
          <div className="rounded-xl border border-border bg-card p-5">
            <p className="text-sm text-muted-foreground">Input</p>
            <p className="mt-2 text-2xl font-semibold">2 Photos</p>
          </div>
          <div className="rounded-xl border border-border bg-card p-5">
            <p className="text-sm text-muted-foreground">Output</p>
            <p className="mt-2 text-2xl font-semibold">High-Quality PNG</p>
          </div>
        </div>
      </Section>

      <Section tone="muted" containerClassName="space-y-8">
        <div className="space-y-3">
          <h2 className="font-display text-4xl leading-tight md:text-5xl">
            Why teams use HairSwap Studio
          </h2>
          <p className="max-w-3xl text-muted-foreground">
            A focused workflow that keeps input, generation, and review in one place.
          </p>
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          {[
            "Fast style exploration before salon sessions",
            "Consistent before/after outputs for portfolio previews",
            "Simple two-image workflow for non-technical users",
            "Proxy API route keeps backend configuration secure",
          ].map((item) => (
            <div key={item} className="flex items-start gap-3 rounded-xl border border-border bg-card p-4">
              <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-accent" />
              <p>{item}</p>
            </div>
          ))}
        </div>
      </Section>

      <Section containerClassName="space-y-5">
        <Badge>Start Now</Badge>
        <h3 className="max-w-3xl font-display text-3xl leading-tight md:text-4xl">
          Ready to create your first swap?
        </h3>
        <div>
          <Link href="/swap">
            <Button variant="secondary">
              <Sparkles className="h-4 w-4" />
              Go to Hair Swap Page
            </Button>
          </Link>
        </div>
      </Section>
    </div>
  );
}
