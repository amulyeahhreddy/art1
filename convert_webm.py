from moviepy.editor import AudioFileClip

def convert_webm_to_wav(input_file, output_file):
    print(f"Converting {input_file} → {output_file}...")
    
    # Load the audio from the webm file
    audio_clip = AudioFileClip(input_file)
    
    # Write to WAV format
    audio_clip.write_audiofile(output_file)
    
    # Close the clip to free resources
    audio_clip.close()
    
    print(f"✅ Done! Saved as {output_file}")

if __name__ == "__main__":
    convert_webm_to_wav("carnatic.webm", "carnatic.wav")
