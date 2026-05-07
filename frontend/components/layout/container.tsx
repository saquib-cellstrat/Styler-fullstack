import * as React from "react";
import { cn } from "@/lib/utils";

export interface ContainerProps extends React.HTMLAttributes<HTMLDivElement> {
  as?: React.ElementType;
}

export function Container({
  as: Comp = "div",
  className,
  ...props
}: ContainerProps) {
  return (
    <Comp
      className={cn("mx-auto w-full max-w-6xl px-[var(--space-container-x)]", className)}
      {...props}
    />
  );
}
