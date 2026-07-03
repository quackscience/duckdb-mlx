#!/usr/bin/env bash
# Diagnose Xcode / Metal toolchain setup for duckdb-mlx builds.
set -euo pipefail

metal_ok=0
if xcrun -sdk macosx metal --version >/dev/null 2>&1; then
	metal_ok=1
fi

echo "=== duckdb-mlx toolchain check ==="
echo ""
echo "Active developer directory (xcode-select -p):"
if devdir="$(xcode-select -p 2>/dev/null)"; then
	echo "  $devdir"
	if [[ "$devdir" == *CommandLineTools* ]]; then
		echo "  ^ Command Line Tools only — Metal compiler is not included."
	fi
else
	echo "  (not set)"
fi
echo ""

echo "Xcode.app search:"
found=()
while IFS= read -r app; do
	[ -n "$app" ] || continue
	found+=("$app")
	echo "  $app"
done < <(mdfind 'kMDItemCFBundleIdentifier == "com.apple.dt.Xcode"' 2>/dev/null | sort -u)
for app in /Applications/Xcode*.app; do
	if [ -d "$app" ]; then
		case " ${found[*]} " in
		*" $app "*) ;;
		*) found+=("$app"); echo "  $app" ;;
		esac
	fi
done
if [ ${#found[@]} -eq 0 ]; then
	echo "  (none — only Command Line Tools or no Xcode installed)"
fi
echo ""

if [ "$metal_ok" -eq 1 ]; then
	echo "OK: xcrun -sdk macosx metal --version"
	xcrun -sdk macosx metal --version
	exit 0
fi

echo "FAIL: Metal compiler not available (required to build vendored MLX)."
echo ""
echo "Homebrew does not ship Apple's Metal compiler. You need the full Xcode.app"
echo "from Apple (App Store or the xcodes CLI), not only:"
echo "  xcode-select --install"
echo "  brew install ...   # no brew formula provides xcrun metal"
echo ""
if [ ${#found[@]} -gt 0 ]; then
	echo "Xcode is installed but not selected. Run ONE of these (copy the full path):"
	for app in "${found[@]}"; do
		dev="$app/Contents/Developer"
		if [ -d "$dev" ]; then
			echo "  sudo xcode-select -s $dev"
		fi
	done
	echo ""
	echo "If you installed via xcodes (brew install xcodesorg/made/xcodes):"
	echo "  xcodes select --latest"
	echo "  xcodes run --latest    # open Xcode once to finish setup"
else
	echo "Install full Xcode (pick one):"
	echo "  App Store → search \"Xcode\" → Install → open once"
	echo "  brew install xcodesorg/made/xcodes"
	echo "  xcodes install --latest"
	echo "  xcodes select --latest"
	echo "  xcodes run --latest"
fi
echo ""
echo "Then accept the license and verify:"
echo "  sudo xcodebuild -license accept"
echo "  xcrun -sdk macosx metal --version"
exit 1
