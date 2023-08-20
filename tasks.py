# A binary builder for Macondo

from invoke import task


@task
def build(c):
    tag = c.run("git describe --exact-match --tags", hide=True).stdout.strip()
    print("Tag was", tag)

    # Build universal mac executable. This only works on Mac:
    c.run(f"GOOS=darwin GOARCH=amd64 go build -o macondo-amd64 ./cmd/shell")
    c.run(f"GOOS=darwin GOARCH=arm64 go build -o macondo-arm64 ./cmd/shell")
    c.run("lipo -create -output macondo macondo-amd64 macondo-arm64")
    c.run(f"zip -r macondo-{tag}-osx-universal.zip ./macondo ./data")

    for os, nickname, arch in [
        ("linux", "linux-x86_64", "amd64"),
        ("windows", "win64", "amd64"),
    ]:
        c.run(f"GOOS={os} GOARCH={arch} go build -o macondo ./cmd/shell")
        c.run(f"zip -r macondo-{tag}-{nickname}.zip ./macondo ./data")
