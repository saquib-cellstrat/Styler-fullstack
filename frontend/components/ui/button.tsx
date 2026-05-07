"use client";

import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "group inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-xl text-sm font-medium transition-all duration-200 ease-out focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:pointer-events-none disabled:opacity-50 active:scale-[0.98]",
  {
    variants: {
      variant: {
        primary:
          "h-12 bg-gradient-to-r from-accent to-accent-secondary px-6 text-accent-foreground shadow-sm hover:-translate-y-0.5 hover:brightness-110",
        secondary:
          "h-12 border border-border bg-transparent px-6 text-foreground hover:-translate-y-0.5 hover:border-accent/30 hover:bg-muted",
        ghost: "h-11 px-4 text-muted-foreground hover:text-foreground",
      },
      size: {
        default: "h-12 px-6",
        sm: "h-10 rounded-lg px-4 text-xs",
        lg: "h-14 px-8 text-base",
      },
      fullWidth: {
        true: "w-full",
        false: "",
      },
    },
    defaultVariants: {
      variant: "primary",
      size: "default",
      fullWidth: false,
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, fullWidth, ...props }, ref) => (
    <button
      className={cn(buttonVariants({ variant, size, fullWidth }), className)}
      ref={ref}
      {...props}
    />
  )
);

Button.displayName = "Button";

export { Button, buttonVariants };
