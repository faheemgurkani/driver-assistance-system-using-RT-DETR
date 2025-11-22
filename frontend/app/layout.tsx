import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Driver Assistance System',
  description: 'Real-time object detection for dashcam videos using RT-DETR',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}

