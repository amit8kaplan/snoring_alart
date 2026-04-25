from audio_utils import copy_and_convert_user_audio_to_wav
from snore_model import predict_snoring_windows


def run_prediction_once():
    user_path = input("\nEnter full path to audio file, or type q to quit: ").strip()

    if user_path.lower() in ["q", "quit", "exit"]:
        return False

    try:
        wav_path = copy_and_convert_user_audio_to_wav(user_path)
        print(f"\nConverted/copied file to: {wav_path}")

        results, summary = predict_snoring_windows(wav_path)

        print("\nWindow results:")
        print("----------------")

        for r in results:
            label = "SNORE" if r["is_snore"] else "not snore"
            print(
                f'{r["start_sec"]:02d}-{r["end_sec"]:02d}s | '
                f'{label:9s} | score={r["snore_score"]:.3f}'
            )

        print("\nSummary:")
        print(f'Total windows: {summary["total_windows"]}')
        print(f'Snore windows: {summary["snore_windows"]}')
        print(f'Snore ratio: {summary["snore_ratio"]:.2%}')

    except Exception as e:
        print("\nError:")
        print(e)

    return True


def main():
    print("Snoring Detection - Audio File Test")
    print("-----------------------------------")
    print("Type q / quit / exit to stop.")

    while True:
        should_continue = run_prediction_once()

        if not should_continue:
            print("Stopped.")
            break


if __name__ == "__main__":
    main()