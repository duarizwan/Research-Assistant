// src/app/layout.tsx
import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Research Assistant",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <style
          dangerouslySetInnerHTML={{
            __html: `
              /* Completely hide NextJS dev tools */
              [data-next-badge="true"],
              [data-nextjs-dev-tools-button="true"],
              [data-issues="true"],
              [data-next-mark="true"],
              [data-issues-open="true"],
              [data-issues-collapse="true"],
              [data-issues-count="true"],
              [data-issues-count-animation="true"],
              [data-issues-count-exit="true"],
              [data-issues-count-enter="true"],
              [data-error="true"],
              [data-error-expanded="true"],
              [data-animate="false"],
              [data-next-mark-loading="false"],
              [data-cross="true"],
          
              div[data-next-badge="true"],
              div[data-issues="true"],
              div[data-issues-count-animation="true"],
              div[data-issues-count="true"],
              div[data-issues-count-exit="true"],
              div[data-issues-count-enter="true"],
              button[data-issues-open="true"],
              button[data-issues-collapse="true"],
              div[data-next-badge],
              div[data-issues],
              div[data-issues-count-animation],
              div[data-issues-count],
              div[data-issues-count-exit],
              div[data-issues-count-enter],
              button[data-issues-open],
              button[data-issues-collapse],
              *[data-next-badge],
              *[data-nextjs-dev-tools-button],
              *[data-issues],
              *[data-next-mark],
              *[data-error],
            
            `,
          }}
        />
      </head>
      <body suppressHydrationWarning>
        <script

        // Remove NextJS dev tools from DOM completely
        />
        {children}
      </body>
    </html>
  );
}
