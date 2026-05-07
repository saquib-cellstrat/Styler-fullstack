import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const cardVariants = cva(
  "rounded-xl border border-border bg-card text-foreground shadow-md transition-all duration-300 ease-out",
  {
    variants: {
      variant: {
        default: "hover:shadow-xl",
        elevated: "shadow-lg hover:-translate-y-1 hover:shadow-xl",
        featured: "bg-gradient-to-br from-accent via-accent-secondary to-accent p-px",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

export interface CardProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof cardVariants> {}

export function Card({ className, variant, ...props }: CardProps) {
  if (variant === "featured") {
    return (
      <div className={cn(cardVariants({ variant }), className)}>
        <div className="h-full w-full rounded-[calc(0.75rem-1px)] bg-card" {...props} />
      </div>
    );
  }

  return <div className={cn(cardVariants({ variant }), className)} {...props} />;
}

export function CardHeader({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("flex flex-col gap-2 p-6", className)} {...props} />;
}

export function CardTitle({
  className,
  ...props
}: React.HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h3
      className={cn("text-xl font-semibold tracking-tight text-foreground", className)}
      {...props}
    />
  );
}

export function CardDescription({
  className,
  ...props
}: React.HTMLAttributes<HTMLParagraphElement>) {
  return <p className={cn("text-sm leading-6 text-muted-foreground", className)} {...props} />;
}

export function CardContent({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("p-6 pt-0", className)} {...props} />;
}
