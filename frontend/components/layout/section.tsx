import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";
import { Container } from "./container";

const sectionVariants = cva("relative py-[var(--space-section-y)]", {
  variants: {
    tone: {
      default: "bg-background text-foreground",
      inverted: "bg-foreground text-background",
      muted: "bg-muted/40 text-foreground",
    },
  },
  defaultVariants: {
    tone: "default",
  },
});

export interface SectionProps
  extends React.HTMLAttributes<HTMLElement>,
    VariantProps<typeof sectionVariants> {
  as?: React.ElementType;
  containerClassName?: string;
}

export function Section({
  as: Comp = "section",
  tone,
  className,
  containerClassName,
  children,
  ...props
}: SectionProps) {
  return (
    <Comp className={cn(sectionVariants({ tone }), className)} {...props}>
      <Container className={containerClassName}>{children}</Container>
    </Comp>
  );
}
