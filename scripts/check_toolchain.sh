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
echo "Homebrew does not ship Apple's Metal compiler. Command Line Tools are not enough."
echo ""
if [ ${#found[@]} -gt 0 ]; then
	echo "Xcode is on disk but not selected. Run ONE of these (copy the full path):"
	for app in "${found[@]}"; do
		dev="$app/Contents/Developer"
		if [ -d "$dev" ]; then
			echo "  sudo xcode-select -s $dev"
		fi
	done
	echo ""
	echo "Open Xcode once from Finder, then:"
else
	echo "Install Xcode.app first (you have none). Pick one:"
	echo ""
	echo "  A) Mac App Store — search \"Xcode\", Install, open once"
	echo "     Or from Terminal (after: brew install mas):"
	echo "       mas install 497799835"
	echo ""
	echo "  B) https://developer.apple.com/download/all/ — download .xip,"
	echo "     double-click to extract, drag Xcode.app to /Applications, open once"
	echo ""
	echo "  Do NOT use 'brew install xcodesorg/made/xcodes' for the first install —"
	echo "  that formula fails without Xcode already present (needs xcbuild)."
	echo ""
	echo "  After Xcode is installed:"
fi
echo "  sudo xcode-select -s /Applications/Xcode.app/Contents/Developer"
echo "  sudo xcodebuild -license accept"
echo "  xcrun -sdk macosx metal --version"
echo ""
echo "  (If Xcode is named Xcode-16.x.app, run this script again for the exact path.)"
exit 1
