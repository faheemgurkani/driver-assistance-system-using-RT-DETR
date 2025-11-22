import { NextResponse } from 'next/server';
import { readFile } from 'fs/promises';
import { join } from 'path';

export async function GET() {
  try {
    // Path to README.md from the project root (frontend -> project root)
    // process.cwd() is the frontend directory, so we go up one level
    const readmePath = join(process.cwd(), '..', 'README.md');
    const content = await readFile(readmePath, 'utf-8');
    return NextResponse.json({ content });
  } catch (error: any) {
    console.error('Error reading README:', error);
    return NextResponse.json(
      { error: 'Failed to read README.md', message: error.message },
      { status: 500 }
    );
  }
}

