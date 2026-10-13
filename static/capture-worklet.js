class VoiceCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(1024);
    this.offset = 0;
  }

  process(inputs, outputs) {
    const input = inputs[0]?.[0];
    if (input) {
      let sourceOffset = 0;
      while (sourceOffset < input.length) {
        const copied = Math.min(
          input.length - sourceOffset,
          this.buffer.length - this.offset,
        );
        this.buffer.set(input.subarray(sourceOffset, sourceOffset + copied), this.offset);
        this.offset += copied;
        sourceOffset += copied;
        if (this.offset === this.buffer.length) {
          this.port.postMessage(this.buffer, [this.buffer.buffer]);
          this.buffer = new Float32Array(1024);
          this.offset = 0;
        }
      }
    }

    for (const channel of outputs[0] ?? []) {
      channel.fill(0);
    }
    return true;
  }
}

registerProcessor("voice-capture", VoiceCaptureProcessor);
