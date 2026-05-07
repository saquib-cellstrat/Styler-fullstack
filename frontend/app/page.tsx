"use client";

import Image from "next/image";
import { motion } from "framer-motion";
import { ArrowRight, BookOpen, Sparkles, WandSparkles } from "lucide-react";
import { Section } from "@/components/layout/section";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";

const easeOut = [0.16, 1, 0.3, 1] as const;

const fadeInUp = {
  hidden: { opacity: 0, y: 28 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.7, ease: easeOut },
  },
};

export default function Home() {
  return (
    <div className="bg-background font-sans text-foreground">
      <Section className="overflow-hidden surface-glow">
        <motion.div
          initial="hidden"
          animate="visible"
          variants={{
            hidden: {},
            visible: { transition: { staggerChildren: 0.1, delayChildren: 0.1 } },
          }}
          className="grid items-center gap-10 lg:grid-cols-[1.1fr_0.9fr]"
        >
          <motion.div variants={fadeInUp} className="space-y-8">
            <Badge pulse>System update</Badge>
            <h1 className="max-w-3xl font-display text-5xl leading-[1.05] tracking-[-0.02em] md:text-7xl">
              Build with clarity, ship with{" "}
              <span className="gradient-text">confident detail</span>
            </h1>
            <p className="max-w-2xl text-lg leading-8 text-muted-foreground">
              A reusable frontend system that centralizes tokens and components while
              preserving behavior. Start from a clean base and scale without style drift.
            </p>
            <motion.div variants={fadeInUp} className="flex flex-col gap-4 sm:flex-row">
              <a
                href="https://vercel.com/new?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
                target="_blank"
                rel="noopener noreferrer"
                className="w-full sm:w-auto"
              >
                <Button className="w-full sm:w-auto">
                  <Image src="/vercel.svg" alt="Vercel logomark" width={16} height={16} />
                  Deploy Now
                  <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
                </Button>
              </a>
              <a
                href="https://nextjs.org/docs?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
                target="_blank"
                rel="noopener noreferrer"
                className="w-full sm:w-auto"
              >
                <Button variant="secondary" className="w-full sm:w-auto">
                  <BookOpen className="h-4 w-4" />
                  Documentation
                </Button>
              </a>
            </motion.div>
          </motion.div>

          <motion.div variants={fadeInUp} className="relative hidden lg:block">
            <div className="relative rounded-2xl border border-border bg-card p-8 shadow-xl">
              <div className="absolute -right-14 -top-14 h-44 w-44 rounded-full border border-accent/25 motion-safe:animate-[slow-spin_60s_linear_infinite]" />
              <Card className="motion-safe:animate-[bob_5s_ease-in-out_infinite]">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <span className="rounded-lg bg-gradient-to-br from-accent to-accent-secondary p-2 text-white">
                      <Sparkles className="h-4 w-4" />
                    </span>
                    Tokenized UI
                  </CardTitle>
                  <CardDescription>
                    Primitive components are now consistent by default.
                  </CardDescription>
                </CardHeader>
              </Card>
              <div className="mt-6 grid grid-cols-3 gap-3">
                {Array.from({ length: 9 }).map((_, index) => (
                  <div
                    key={index}
                    className="aspect-square rounded-lg bg-gradient-to-br from-accent/15 to-accent-secondary/10"
                  />
                ))}
              </div>
            </div>
          </motion.div>
        </motion.div>
      </Section>

      <Section tone="inverted" className="dot-grid">
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.15, margin: "-60px" }}
          variants={{
            hidden: {},
            visible: { transition: { staggerChildren: 0.1, delayChildren: 0.1 } },
          }}
          className="space-y-8"
        >
          <motion.div variants={fadeInUp}>
            <Badge variant="inverse" pulse>
              Implementation
            </Badge>
          </motion.div>
          <motion.h2
            variants={fadeInUp}
            className="font-display text-4xl leading-tight md:text-5xl"
          >
            Consistent architecture from{" "}
            <span className="gradient-text">tokens to pages</span>
          </motion.h2>
          <motion.div variants={fadeInUp} className="grid gap-5 md:grid-cols-3">
            <Card className="bg-white/5 text-background shadow-lg backdrop-blur">
              <CardHeader>
                <CardTitle>Reusable Primitives</CardTitle>
                <CardDescription className="text-slate-200">
                  Button, Card, Badge, Input, Section, and Container share one visual language.
                </CardDescription>
              </CardHeader>
            </Card>
            <Card className="bg-white/5 text-background shadow-lg backdrop-blur">
              <CardHeader>
                <CardTitle>Centralized Tokens</CardTitle>
                <CardDescription className="text-slate-200">
                  Colors, radii, shadows, and typography are semantic and predictable.
                </CardDescription>
              </CardHeader>
            </Card>
            <Card className="bg-white/5 text-background shadow-lg backdrop-blur">
              <CardHeader>
                <CardTitle>Motion Discipline</CardTitle>
                <CardDescription className="text-slate-200">
                  Motion is purposeful, subtle, and respects reduced-motion preferences.
                </CardDescription>
              </CardHeader>
            </Card>
          </motion.div>
        </motion.div>
      </Section>

      <Section>
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.2 }}
          variants={{
            hidden: {},
            visible: { transition: { staggerChildren: 0.1, delayChildren: 0.1 } },
          }}
          className="space-y-8"
        >
          <motion.div variants={fadeInUp}>
            <Badge>Stay Updated</Badge>
          </motion.div>
          <motion.h2 variants={fadeInUp} className="font-display text-4xl md:text-5xl">
            Join the <span className="gradient-text">build log</span>
          </motion.h2>
          <motion.div variants={fadeInUp}>
            <Card variant="featured">
              <CardContent className="space-y-4 p-6 md:p-8">
                <p className="text-muted-foreground">
                  Get release notes and architecture updates as the system evolves.
                </p>
                <div className="flex flex-col gap-3 sm:flex-row">
                  <Input type="email" placeholder="name@company.com" className="h-12" />
                  <Button className="h-12 sm:w-auto">
                    <WandSparkles className="h-4 w-4" />
                    Subscribe
                  </Button>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </motion.div>
      </Section>
    </div>
  );
}
