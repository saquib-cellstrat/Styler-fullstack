import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-2 rounded-full border px-3 py-1 font-mono text-xs uppercase tracking-[0.15em]",
  {
    variants: {
      variant: {
        default: "border-accent/30 bg-accent/5 text-accent",
        neutral: "border-border bg-muted text-muted-foreground",
        inverse: "border-white/20 bg-white/10 text-white",
      },
      pulse: {
        true: "",
        false: "",
      },
    },
    defaultVariants: {
      variant: "default",
      pulse: false,
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {
  dot?: boolean;
}

export function Badge({
  className,
  variant,
  pulse,
  dot = true,
  children,
  ...props
}: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant, pulse }), className)} {...props}>
      {dot ? (
        <span
          className={cn(
            "h-2 w-2 rounded-full bg-current",
            pulse && "motion-safe:animate-[pulse-dot_2s_ease-in-out_infinite]"
          )}
          aria-hidden
        />
      ) : null}
      <span>{children}</span>
    </div>
  );
}
