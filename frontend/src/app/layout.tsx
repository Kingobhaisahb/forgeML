import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ForgeML | Model Training Platform",
  description:
    "Train, evaluate, and manage machine learning models with ForgeML.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}