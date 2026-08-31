package main

import (
	"bytes"
	"flag"
	"fmt"
	"image"
	"image/draw"
	"image/png"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type Config struct {
	RootDir    string
	PDFWorkers int
	Workers    int
	DPI        int
	CropX1     int
	CropY1     int
	CropX2     int
	CropY2     int
	Lang       string
	Verbose    bool
}

type PDFTask struct {
	ID         int
	PDFPath    string
	BaseName   string
	ParentDir  string
	ImagesDir  string
	MDDir      string
	PagesMDDir string
}

type PageResult struct {
	PageNum  int
	FileName string
	Text     string
}

type SubImager interface {
	SubImage(r image.Rectangle) image.Image
}

func main() {
	var cfg Config
	flag.StringVar(&cfg.RootDir, "dir", ".", "Directory to scan for PDF files")
	flag.IntVar(&cfg.PDFWorkers, "pdf-workers", 2, "Number of PDF files to convert concurrently")
	flag.IntVar(&cfg.Workers, "workers", runtime.NumCPU(), "Number of parallel workers")
	flag.IntVar(&cfg.DPI, "dpi", 300, "Resolution (DPI)")
	flag.IntVar(&cfg.CropX1, "x1", 154, "Crop X1")
	flag.IntVar(&cfg.CropY1, "y1", 236, "Crop Y1")
	flag.IntVar(&cfg.CropX2, "x2", 2392, "Crop X2")
	flag.IntVar(&cfg.CropY2, "y2", 3007, "Crop Y2")
	flag.StringVar(&cfg.Lang, "lang", "eng", "OCR language")
	flag.BoolVar(&cfg.Verbose, "v", false, "Verbose")
	flag.Parse()

	absRoot, _ := filepath.Abs(cfg.RootDir)
	cfg.RootDir = absRoot

	fmt.Println("================================================================================")
	fmt.Println("         MASSIVE PARALLEL PDF TO CROPPED MARKDOWN CONVERTER (CPU)")
	fmt.Println("================================================================================")
	fmt.Printf("[*] Root: %s | Workers: %d | DPI: %d\n", cfg.RootDir, cfg.Workers, cfg.DPI)
	fmt.Println("--------------------------------------------------------------------------------")

	checkDeps()
	startTime := time.Now()

	pdfTasks, _ := discoverPDFs(cfg.RootDir)
	if len(pdfTasks) == 0 {
		fmt.Printf("[!] No PDF files in %s\n", cfg.RootDir)
		return
	}

	fmt.Printf("[+] Found %d PDF(s)\n\n", len(pdfTasks))

	var rendered, cropped, ocr int64
	for i, task := range pdfTasks {
		processPDF(cfg, i+1, len(pdfTasks), task, &rendered, &cropped, &ocr)
	}

	elapsed := time.Since(startTime)
	fmt.Printf("\nCOMPLETED: %d PDFs, %d pages in %v\n", len(pdfTasks), ocr, elapsed.Round(time.Second))
}

func checkDeps() {
	for _, t := range []string{"pdftoppm", "tesseract"} {
		if _, err := exec.LookPath(t); err != nil {
			fmt.Fprintf(os.Stderr, "[!] Missing: %s\n", t)
			os.Exit(1)
		}
	}
}

func discoverPDFs(root string) ([]*PDFTask, error) {
	var tasks []*PDFTask
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		if !strings.EqualFold(filepath.Ext(path), ".pdf") {
			return nil
		}
		base := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
		parentDir := filepath.Dir(path)
		relParent, _ := filepath.Rel(root, parentDir)
		if relParent == "." {
			relParent = filepath.Base(root)
		}
		tasks = append(tasks, &PDFTask{
			ID:         len(tasks) + 1,
			PDFPath:    path,
			BaseName:   base,
			ParentDir:  parentDir,
			ImagesDir:  filepath.Join(parentDir, base, "images"),
			MDDir:      filepath.Join(parentDir, base+"-md"),
			PagesMDDir: filepath.Join(filepath.Join(parentDir, base+"-md"), "pages_md"),
		})
		return nil
	})
	return tasks, err
}

func processPDF(cfg Config, idx, total int, task *PDFTask, rendered, cropped, ocr *int64) {
	fmt.Printf("[%d/%d] %s\n", idx, total, task.BaseName)

	os.MkdirAll(task.ImagesDir, 0755)
	os.MkdirAll(task.PagesMDDir, 0755)

	fmt.Print("  Render...")
	pngFiles, _ := renderPDF(cfg, task)
	atomic.AddInt64(rendered, int64(len(pngFiles)))
	fmt.Printf(" %d pages\n", len(pngFiles))

	fmt.Print("  Crop...")
	croppedCount := cropImages(cfg, pngFiles)
	atomic.AddInt64(cropped, int64(croppedCount))
	fmt.Printf(" %d\n", croppedCount)

	fmt.Print("  OCR...")
	ocrResults := ocrImages(task, pngFiles, cfg)
	atomic.AddInt64(ocr, int64(len(ocrResults)))
	fmt.Printf(" %d pages\n", len(ocrResults))

	combineMD(task.BaseName, task.MDDir, ocrResults)
	fmt.Printf("  [done in %v]\n\n", time.Since(time.Now()).Round(time.Second))
}

func renderPDF(cfg Config, task *PDFTask) ([]string, error) {
	prefix := filepath.Join(task.ImagesDir, "page")
	cmd := exec.Command("pdftoppm", "-png", "-r", strconv.Itoa(cfg.DPI), task.PDFPath, prefix)
	cmd.Run()
	files, _ := filepath.Glob(filepath.Join(task.ImagesDir, "page-*.png"))
	sortImagePaths(files)
	return files, nil
}

func cropImages(cfg Config, pngFiles []string) int {
	cropRect := image.Rect(cfg.CropX1, cfg.CropY1, cfg.CropX2+1, cfg.CropY2+1)

	type job struct{ path string }
	jobs := make(chan job, len(pngFiles))
	for _, p := range pngFiles {
		jobs <- job{path: p}
	}
	close(jobs)

	var success int64
	var wg sync.WaitGroup
	for w := 0; w < cfg.Workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				if err := cropFile(j.path, cropRect); err == nil {
					atomic.AddInt64(&success, 1)
				}
			}
		}()
	}
	wg.Wait()
	return int(success)
}

func cropFile(path string, rect image.Rectangle) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	img, err := png.Decode(file)
	file.Close()
	if err != nil {
		return err
	}

	bounds := img.Bounds()
	actualCrop := bounds.Intersect(rect)
	if actualCrop.Dx() == 0 || actualCrop.Dy() == 0 {
		return nil
	}

	dst := image.NewRGBA(image.Rect(0, 0, actualCrop.Dx(), actualCrop.Dy()))
	draw.Draw(dst, dst.Bounds(), img, actualCrop.Min, draw.Src)

	tmp := path + ".tmp"
	out, _ := os.Create(tmp)
	png.Encode(out, dst)
	out.Close()
	return os.Rename(tmp, path)
}

func ocrImages(task *PDFTask, pngFiles []string, cfg Config) []*PageResult {
	rePage := regexp.MustCompile(`page-0*(\d+)\.png`)

	type job struct {
		pageNum  int
		filePath string
	}
	jobs := make(chan job, len(pngFiles))
	for i, p := range pngFiles {
		pn := i + 1
		if m := rePage.FindStringSubmatch(filepath.Base(p)); len(m) > 1 {
			if n, _ := strconv.Atoi(m[1]); n > 0 {
				pn = n
			}
		}
		jobs <- job{pageNum: pn, filePath: p}
	}
	close(jobs)

	var mu sync.Mutex
	var results []*PageResult
	var wg sync.WaitGroup

	for w := 0; w < cfg.Workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				text, _ := runTesseract(j.filePath, cfg.Lang)
				md := fmt.Sprintf("# Page %d\n\n![Page](images/%s)\n\n%s\n---\n", j.pageNum, filepath.Base(j.filePath), strings.TrimSpace(text))
				mdPath := filepath.Join(task.PagesMDDir, fmt.Sprintf("page_%03d.md", j.pageNum))
				os.WriteFile(mdPath, []byte(md), 0644)

				mu.Lock()
				results = append(results, &PageResult{PageNum: j.pageNum, FileName: filepath.Base(j.filePath), Text: text})
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	sort.Slice(results, func(i, j int) bool { return results[i].PageNum < results[j].PageNum })
	return results
}

func runTesseract(path, lang string) (string, error) {
	cmd := exec.Command("tesseract", path, "stdout", "-l", lang)
	var out bytes.Buffer
	cmd.Stdout = &out
	return out.String(), cmd.Run()
}

func combineMD(title, mdDir string, pages []*PageResult) {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("# %s\n\n> %d pages\n\n---\n\n", title, len(pages)))
	for _, p := range pages {
		sb.WriteString(fmt.Sprintf("## Page %d\n\n![Page](images/%s)\n\n%s\n\n---\n\n", p.PageNum, p.FileName, strings.TrimSpace(p.Text)))
	}
	os.WriteFile(filepath.Join(mdDir, title+".md"), []byte(sb.String()), 0644)
}

func sortImagePaths(paths []string) {
	re := regexp.MustCompile(`page-0*(\d+)\.png`)
	sort.Slice(paths, func(i, j int) bool {
		m1, m2 := re.FindStringSubmatch(filepath.Base(paths[i])), re.FindStringSubmatch(filepath.Base(paths[j]))
		if len(m1) > 1 && len(m2) > 1 {
			n1, _ := strconv.Atoi(m1[1])
			n2, _ := strconv.Atoi(m2[1])
			return n1 < n2
		}
		return paths[i] < paths[j]
	})
}