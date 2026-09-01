package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/draw"
	"image/png"
	"io"
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
	"unicode"
)

// Config holds all user parameters
type Config struct {
	RootDir       string
	PDFWorkers    int
	Workers       int
	DPI           int
	CropX1        int
	CropY1        int
	CropX2        int
	CropY2        int
	Lang          string
	Verbose       bool
	Format        string // Output format: md, txt, json
	Quality       int    // JPEG quality (1-100), 0 = PNG
	Checkpoint    string // Path to checkpoint file for resume
	NoCleanup     bool   // Keep intermediate files
	SkipEmpty     bool   // Skip pages with no extracted text
	StripHeader   bool   // Remove header and image from individual page files
	Pretty        bool   // Format text for better readability
	UseGPU        bool   // Enable GPU acceleration
}

var gpu *GPUProcessor

func initGPU() {
	if gpu == nil {
		gpu = NewGPUProcessor()
	}
}

type PDFTask struct {
	ID         int
	PDFPath    string
	RelPath    string
	BaseName   string
	ParentDir  string
	ImagesDir  string
	MDDir      string
	PagesMDDir string
	Pages      []string
}

type SubImager interface {
	SubImage(r image.Rectangle) image.Image
}

type PageResult struct {
	PageNum  int
	FileName string
	Text     string
}

type Checkpoint struct {
	ProcessedPDFs []string            `json:"processed_pdfs"`
	ProcessedPages map[string][]int   `json:"processed_pages"` // pdf_base_name -> page numbers
	LastUpdated    time.Time          `json:"last_updated"`
}

// Pre-compiled regex patterns (compiled once at package level)
var (
	rePageNum    = regexp.MustCompile(`page-0*(\d+)\.png`)
	rePageNumSort = regexp.MustCompile(`page-0*(\d+)\.png`)
)

func main() {
	var cfg Config
	flag.StringVar(&cfg.RootDir, "dir", ".", "Directory to scan for PDF files")
	flag.IntVar(&cfg.PDFWorkers, "pdf-workers", 2, "Number of PDF files to process concurrently")
	flag.IntVar(&cfg.Workers, "workers", runtime.NumCPU(), "Number of parallel workers for Cropping & OCR")
	flag.IntVar(&cfg.DPI, "dpi", 300, "Resolution (DPI) for rendering PDF pages to PNG")
	flag.IntVar(&cfg.CropX1, "x1", 154, "Crop bounding box top-left X coordinate")
	flag.IntVar(&cfg.CropY1, "y1", 236, "Crop bounding box top-left Y coordinate")
	flag.IntVar(&cfg.CropX2, "x2", 2392, "Crop bounding box bottom-right X coordinate")
	flag.IntVar(&cfg.CropY2, "y2", 3007, "Crop bounding box bottom-right Y coordinate")
	flag.StringVar(&cfg.Lang, "lang", "eng", "Tesseract OCR language")
	flag.BoolVar(&cfg.Verbose, "v", false, "Enable verbose logging")
	flag.StringVar(&cfg.Format, "format", "md", "Output format: md, txt, json")
	flag.IntVar(&cfg.Quality, "quality", 0, "JPEG quality 1-100 (0=PNG, default)")
	flag.StringVar(&cfg.Checkpoint, "checkpoint", "", "Path to checkpoint file for resume support")
	flag.BoolVar(&cfg.NoCleanup, "no-cleanup", false, "Keep intermediate image files")
	flag.BoolVar(&cfg.SkipEmpty, "skip-empty", false, "Skip pages with no extracted text (delete individual page file)")
	flag.BoolVar(&cfg.StripHeader, "strip-header", false, "Remove header and image from individual page files")
	flag.BoolVar(&cfg.Pretty, "pretty", false, "Format text for better readability (join lines, fix bullets)")
	flag.BoolVar(&cfg.UseGPU, "gpu", true, "Enable GPU acceleration (CUDA)")
	flag.Parse()

	absRoot, err := filepath.Abs(cfg.RootDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error resolving directory %s: %v\n", cfg.RootDir, err)
		os.Exit(1)
	}
	cfg.RootDir = absRoot

	// Initialize GPU
	initGPU()

	fmt.Println("================================================================================")
	fmt.Println("          OPTIMIZED PARALLEL PDF TO MARKDOWN CONVERTER (CUDA)                  ")
	fmt.Println("================================================================================")
	fmt.Printf("[*] Root Directory     : %s\n", cfg.RootDir)
	fmt.Printf("[*] CPU Cores / Workers: %d (PDF Concurrent: %d, Page Workers: %d)\n", runtime.NumCPU(), cfg.PDFWorkers, cfg.Workers)
	fmt.Printf("[*] Render DPI         : %d DPI\n", cfg.DPI)
	fmt.Printf("[*] Crop Coordinates   : (%d, %d) -> (%d, %d) [Size: %dx%d px]\n",
		cfg.CropX1, cfg.CropY1, cfg.CropX2, cfg.CropY2, cfg.CropX2-cfg.CropX1+1, cfg.CropY2-cfg.CropY1+1)
	fmt.Printf("[*] OCR Language       : %s\n", cfg.Lang)
	fmt.Printf("[*] Output Format      : %s\n", cfg.Format)
	if cfg.Quality > 0 {
		fmt.Printf("[*] Image Quality      : %d (JPEG)\n", cfg.Quality)
	}
	fmt.Println("--------------------------------------------------------------------------------")

	checkDependencies()

	startTime := time.Now()

	// Load checkpoint if exists
	var checkpoint *Checkpoint
	if cfg.Checkpoint != "" {
		checkpoint = loadCheckpoint(cfg.Checkpoint)
		fmt.Printf("[*] Resuming from checkpoint: %s (%d PDFs already processed)\n", cfg.Checkpoint, len(checkpoint.ProcessedPDFs))
	}

	// STEP 1: Discovery
	fmt.Println("\n[PHASE 1] Scanning directory for PDF files...")
	pdfTasks, folderMap := discoverPDFs(cfg.RootDir)

	if len(pdfTasks) == 0 {
		fmt.Printf("[*] No PDF files found in %s\n", cfg.RootDir)
		return
	}

	// Filter out already processed PDFs if resuming
	if checkpoint != nil && len(checkpoint.ProcessedPDFs) > 0 {
		var remaining []*PDFTask
		processedSet := make(map[string]bool, len(checkpoint.ProcessedPDFs))
		for _, p := range checkpoint.ProcessedPDFs {
			processedSet[p] = true
		}
		for _, task := range pdfTasks {
			if !processedSet[task.BaseName] {
				remaining = append(remaining, task)
			}
		}
		fmt.Printf("[*] Skipping %d already processed PDFs\n", len(pdfTasks)-len(remaining))
		pdfTasks = remaining
	}

	if len(pdfTasks) == 0 {
		fmt.Println("[*] All PDFs already processed!")
		return
	}

	fmt.Printf("[+] Discovered %d PDF file(s) across %d folder(s)\n", len(pdfTasks), len(folderMap))

	// STEP 2-6: Process PDFs concurrently
	totalPDFs := len(pdfTasks)
	var totalPagesRendered int64
	var totalPagesCropped int64
	var totalPagesOCR int64

	fmt.Printf("\n[PHASE 2-5] Processing %d PDF(s) with %d concurrent PDF workers...\n\n", totalPDFs, cfg.PDFWorkers)

	// Semaphore for limiting concurrent PDFs
	sem := make(chan struct{}, cfg.PDFWorkers)
	var wg sync.WaitGroup
	var mu sync.Mutex

	for i, task := range pdfTasks {
		wg.Add(1)
		sem <- struct{}{}

		go func(idx int, t *PDFTask) {
			defer wg.Done()
			defer func() { <-sem }()

			pdfStartTime := time.Now()
			fmt.Printf("[%d/%d] Processing: %s\n", idx+1, totalPDFs, t.BaseName)

			if err := processSinglePDF(t, cfg, checkpoint, &totalPagesRendered, &totalPagesCropped, &totalPagesOCR); err != nil {
				fmt.Printf("  [!] Error processing %s: %v\n", t.BaseName, err)
				return
			}

			fmt.Printf("  [✓] Completed in %v\n", time.Since(pdfStartTime).Round(time.Millisecond))

			// Update checkpoint
			if cfg.Checkpoint != "" {
				mu.Lock()
				checkpoint.ProcessedPDFs = append(checkpoint.ProcessedPDFs, t.BaseName)
				checkpoint.LastUpdated = time.Now()
				saveCheckpoint(cfg.Checkpoint, checkpoint)
				mu.Unlock()
			}
		}(i, task)
	}

	wg.Wait()

	// Final Summary
	elapsed := time.Since(startTime)
	fmt.Println("\n================================================================================")
	fmt.Println("                       PROCESSING COMPLETED SUMMARY                            ")
	fmt.Println("================================================================================")
	fmt.Printf("Total PDFs Processed   : %d\n", totalPDFs)
	fmt.Printf("Total Pages Rendered   : %d\n", totalPagesRendered)
	fmt.Printf("Total Pages Cropped    : %d\n", totalPagesCropped)
	fmt.Printf("Total Pages OCR        : %d\n", totalPagesOCR)
	fmt.Printf("Total Elapsed Time     : %v\n", elapsed.Round(time.Millisecond))
	if totalPagesOCR > 0 && elapsed.Seconds() > 0 {
		fmt.Printf("Overall Speed          : %.2f pages/sec\n", float64(totalPagesOCR)/elapsed.Seconds())
	}
	fmt.Println("================================================================================")
}

func processSinglePDF(task *PDFTask, cfg Config, checkpoint *Checkpoint, totalRendered, totalCropped, totalOCR *int64) error {
	// Create directories
	if err := os.MkdirAll(task.ImagesDir, 0755); err != nil {
		return fmt.Errorf("create image dir: %w", err)
	}
	if err := os.MkdirAll(task.PagesMDDir, 0755); err != nil {
		return fmt.Errorf("create MD dir: %w", err)
	}

	// STEP 2: Render PDF to PNG
	pngFiles, err := renderPDFToPNG(task.PDFPath, task.ImagesDir, cfg.DPI, cfg.Verbose)
	if err != nil {
		return fmt.Errorf("render PDF: %w", err)
	}
	atomic.AddInt64(totalRendered, int64(len(pngFiles)))

	// STEP 3: Crop (GPU or CPU)
	if cfg.UseGPU && gpu != nil && gpu.Available() {
		fmt.Printf("  -> [Step 3] Cropping with GPU acceleration...\n")
		croppedCount := cropImagesGPU(pngFiles, cfg, gpu)
		atomic.AddInt64(totalCropped, int64(croppedCount))
	} else {
		fmt.Printf("  -> [Step 3] Cropping with CPU...\n")
		croppedCount := cropImagesParallel(pngFiles, cfg, cfg.Workers)
		atomic.AddInt64(totalCropped, int64(croppedCount))
	}

	// STEP 4 & 5: Parallel OCR
	ocrResults := ocrToMarkdownParallel(task, pngFiles, cfg, cfg.Workers, checkpoint)
	atomic.AddInt64(totalOCR, int64(len(ocrResults)))

	// Generate combined output
	switch cfg.Format {
	case "json":
		err = generateJSONOutput(task, ocrResults, cfg.SkipEmpty)
	case "txt":
		err = generateTextOutput(task, ocrResults, cfg.SkipEmpty)
	default:
		err = generateCombinedMarkdown(task.BaseName, filepath.Join(task.MDDir, task.BaseName+".md"), ocrResults, cfg.SkipEmpty)
	}
	if err != nil {
		return fmt.Errorf("generate output: %w", err)
	}

	// Cleanup intermediate files if requested
	if !cfg.NoCleanup {
		// Keep cropped images, remove originals if different
	}

	return nil
}

func checkDependencies() {
	tools := []string{"pdftoppm", "tesseract"}
	missing := make([]string, 0, len(tools))
	for _, tool := range tools {
		if _, err := exec.LookPath(tool); err != nil {
			missing = append(missing, tool)
		}
	}
	if len(missing) > 0 {
		fmt.Fprintf(os.Stderr, "[!] Missing required system tool(s): %s\n", strings.Join(missing, ", "))
		fmt.Fprintf(os.Stderr, "    Install with: sudo apt-get install poppler-utils tesseract-ocr\n")
		os.Exit(1)
	}
}

func discoverPDFs(rootDir string) ([]*PDFTask, map[string][]*PDFTask) {
	tasks := make([]*PDFTask, 0)
	folderMap := make(map[string][]*PDFTask)
	id := 0

	// Use WalkDir which is more efficient than Walk (no os.Stat calls)
	filepath.WalkDir(rootDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil // Skip inaccessible files
		}
		if d.IsDir() {
			return nil
		}
		if !strings.EqualFold(filepath.Ext(d.Name()), ".pdf") {
			return nil
		}

		id++
		base := strings.TrimSuffix(d.Name(), filepath.Ext(d.Name()))
		parentDir := filepath.Dir(path)
		relParent, _ := filepath.Rel(rootDir, parentDir)
		if relParent == "." {
			relParent = filepath.Base(rootDir)
		}

		imagesDir := filepath.Join(parentDir, base, "images")
		mdDir := filepath.Join(parentDir, base+"-md")
		pagesMDDir := filepath.Join(mdDir, "pages_md")

		task := &PDFTask{
			ID:         id,
			PDFPath:    path,
			RelPath:    path,
			BaseName:   base,
			ParentDir:  parentDir,
			ImagesDir:  imagesDir,
			MDDir:      mdDir,
			PagesMDDir: pagesMDDir,
		}
		tasks = append(tasks, task)
		folderMap[relParent] = append(folderMap[relParent], task)
		return nil
	})

	return tasks, folderMap
}

func renderPDFToPNG(pdfPath, outDir string, dpi int, verbose bool) ([]string, error) {
	prefix := filepath.Join(outDir, "page")
	cmd := exec.Command("pdftoppm", "-png", "-r", strconv.Itoa(dpi), pdfPath, prefix)
	if verbose {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}
	if err := cmd.Run(); err != nil {
		return nil, err
	}

	// Use Glob with specific pattern for efficiency
	files, err := filepath.Glob(filepath.Join(outDir, "page-*.png"))
	if err != nil || len(files) == 0 {
		files, _ = filepath.Glob(filepath.Join(outDir, "*.png"))
	}

	sortImagePaths(files)
	return files, nil
}

func cropImagesParallel(pngFiles []string, cfg Config, numWorkers int) int {
	cropRect := image.Rect(cfg.CropX1, cfg.CropY1, cfg.CropX2+1, cfg.CropY2+1)
	expectedW := cropRect.Dx()
	expectedH := cropRect.Dy()

	jobs := make(chan string, len(pngFiles))
	for _, p := range pngFiles {
		jobs <- p
	}
	close(jobs)

	var successCount int64
	var wg sync.WaitGroup

	// Limit workers to file count
	if numWorkers > len(pngFiles) {
		numWorkers = len(pngFiles)
	}

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for path := range jobs {
				if err := cropSingleImage(path, cropRect, expectedW, expectedH); err == nil {
					atomic.AddInt64(&successCount, 1)
				} else if cfg.Verbose {
					fmt.Printf("     [!] Crop error on %s: %v\n", filepath.Base(path), err)
				}
			}
		}()
	}

	wg.Wait()
	return int(successCount)
}

func cropImagesGPU(pngFiles []string, cfg Config, gpu *GPUProcessor) int {
	if !gpu.Available() {
		fmt.Printf("     [GPU] Not available, falling back to CPU\n")
		return cropImagesParallel(pngFiles, cfg, cfg.Workers)
	}

	fmt.Printf("     [GPU] Processing %d images on %s...\n", len(pngFiles), gpu.Info())

	cropRect := image.Rect(cfg.CropX1, cfg.CropY1, cfg.CropX2+1, cfg.CropY2+1)
	expectedW := cropRect.Dx()
	expectedH := cropRect.Dy()

	var successCount int64
	var wg sync.WaitGroup

	/* Use worker pool for GPU processing */
	numWorkers := cfg.Workers
	if numWorkers > 4 {
		numWorkers = 4 /* GPU works better with fewer concurrent streams */
	}

	jobs := make(chan string, len(pngFiles))
	for _, p := range pngFiles {
		jobs <- p
	}
	close(jobs)

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for path := range jobs {
				if err := cropSingleImageGPU(path, cropRect, expectedW, expectedH); err == nil {
					atomic.AddInt64(&successCount, 1)
				} else if cfg.Verbose {
					fmt.Printf("     [!] GPU crop error on %s: %v\n", filepath.Base(path), err)
				}
			}
		}(w)
	}

	wg.Wait()
	return int(successCount)
}

func cropSingleImage(filePath string, targetRect image.Rectangle, targetW, targetH int) error {
	file, err := os.Open(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	img, err := png.Decode(file)
	if err != nil {
		return err
	}

	bounds := img.Bounds()
	// Skip if already cropped to target size
	if bounds.Dx() == targetW && bounds.Dy() == targetH && bounds.Min.X == 0 && bounds.Min.Y == 0 {
		return nil
	}

	actualCropRect := bounds.Intersect(targetRect)
	var cropped image.Image

	if sub, ok := img.(SubImager); ok {
		cropped = sub.SubImage(actualCropRect)
	} else {
		dst := image.NewRGBA(image.Rect(0, 0, actualCropRect.Dx(), actualCropRect.Dy()))
		draw.Draw(dst, dst.Bounds(), img, actualCropRect.Min, draw.Src)
		cropped = dst
	}

	// Write to temp file then rename for atomicity
	tmpFile := filePath + ".tmp"
	outFile, err := os.Create(tmpFile)
	if err != nil {
		return err
	}

	// Use buffered writer for better I/O performance
	bufWriter := bufio.NewWriter(outFile)
	enc := png.Encoder{CompressionLevel: png.DefaultCompression}
	if err := enc.Encode(bufWriter, cropped); err != nil {
		outFile.Close()
		os.Remove(tmpFile)
		return err
	}
	if err := bufWriter.Flush(); err != nil {
		outFile.Close()
		os.Remove(tmpFile)
		return err
	}
	if err := outFile.Close(); err != nil {
		os.Remove(tmpFile)
		return err
	}

	return os.Rename(tmpFile, filePath)
}

/* GPU-accelerated image crop */
func cropSingleImageGPU(filePath string, targetRect image.Rectangle, targetW, targetH int) error {
	file, err := os.Open(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	img, err := png.Decode(file)
	if err != nil {
		return err
	}

	bounds := img.Bounds()
	// Skip if already cropped to target size
	if bounds.Dx() == targetW && bounds.Dy() == targetH && bounds.Min.X == 0 && bounds.Min.Y == 0 {
		return nil
	}

	actualCropRect := bounds.Intersect(targetRect)
	cropW := actualCropRect.Dx()
	cropH := actualCropRect.Dy()
	if cropW <= 0 || cropH <= 0 {
		return fmt.Errorf("invalid crop dimensions")
	}

	// Get raw pixel data
	var rawData []byte
	switch src := img.(type) {
	case *image.RGBA:
		rawData = src.Pix
	case *image.NRGBA:
		rawData = src.Pix
	default:
		// Convert to RGBA
		rgba := image.NewRGBA(bounds)
		draw.Draw(rgba, bounds, img, bounds.Min, draw.Src)
		rawData = rgba.Pix
	}

	// Allocate output buffer
	channels := 4 // RGBA
	dstSize := cropW * cropH * channels
	dstData := make([]byte, dstSize)

	// Call GPU crop
	ret := gpuCropImage(rawData, bounds.Dx(), bounds.Dy(), channels, dstData,
		actualCropRect.Min.X, actualCropRect.Min.Y,
		actualCropRect.Min.X+cropW, actualCropRect.Min.Y+cropH)
	if ret != 0 {
		return fmt.Errorf("GPU crop failed")
	}

	// Create cropped image
	cropped := &image.RGBA{
		Pix:    dstData,
		Stride: cropW * channels,
		Rect:   image.Rect(0, 0, cropW, cropH),
	}

	// Write to temp file then rename for atomicity
	tmpFile := filePath + ".tmp"
	outFile, err := os.Create(tmpFile)
	if err != nil {
		return err
	}

	bufWriter := bufio.NewWriter(outFile)
	enc := png.Encoder{CompressionLevel: png.DefaultCompression}
	if err := enc.Encode(bufWriter, cropped); err != nil {
		outFile.Close()
		os.Remove(tmpFile)
		return err
	}
	if err := bufWriter.Flush(); err != nil {
		outFile.Close()
		os.Remove(tmpFile)
		return err
	}
	if err := outFile.Close(); err != nil {
		os.Remove(tmpFile)
		return err
	}

	return os.Rename(tmpFile, filePath)
}

func ocrToMarkdownParallel(task *PDFTask, pngFiles []string, cfg Config, numWorkers int, checkpoint *Checkpoint) []*PageResult {
	// Determine which pages need processing
	processedPages := make(map[int]bool)
	if checkpoint != nil && checkpoint.ProcessedPages != nil {
		if pages, ok := checkpoint.ProcessedPages[task.BaseName]; ok {
			for _, p := range pages {
				processedPages[p] = true
			}
		}
	}

	type ocrJob struct {
		Index    int
		PageNum  int
		FilePath string
	}

	// Filter out already processed pages
	jobs := make(chan ocrJob, len(pngFiles))
	for i, p := range pngFiles {
		pageNum := extractPageNum(p, i+1)
		if !processedPages[pageNum] {
			jobs <- ocrJob{Index: i, PageNum: pageNum, FilePath: p}
		}
	}
	close(jobs)

	var mu sync.Mutex
	results := make([]*PageResult, 0, len(pngFiles))
	var wg sync.WaitGroup

	// Limit workers
	if numWorkers > len(pngFiles) {
		numWorkers = len(pngFiles)
	}
	if numWorkers < 1 {
		numWorkers = 1
	}

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for job := range jobs {
				text, err := runTesseract(job.FilePath, cfg.Lang)
				if err != nil {
					text = fmt.Sprintf("[OCR Error: %v]", err)
				}

				trimmedText := strings.TrimSpace(text)
				isEmpty := trimmedText == "" || strings.HasPrefix(trimmedText, "[OCR Error:")

				// Skip empty/error pages if flag is set
				if cfg.SkipEmpty && isEmpty {
					mu.Lock()
					results = append(results, &PageResult{
						PageNum:  job.PageNum,
						FileName: filepath.Base(job.FilePath),
						Text:     "",
					})
					mu.Unlock()

					// Remove the file if it was already written with error
					pageFile := filepath.Join(task.PagesMDDir, fmt.Sprintf("page_%03d.md", job.PageNum))
					os.Remove(pageFile)
					continue
				}

				// Write individual page file
				if err := writePageOutput(task, job.PageNum, job.FilePath, text, cfg); err != nil && cfg.Verbose {
					fmt.Printf("     [!] Write error: %v\n", err)
				}

				mu.Lock()
				results = append(results, &PageResult{
					PageNum:  job.PageNum,
					FileName: filepath.Base(job.FilePath),
					Text:     text,
				})
				mu.Unlock()

				// Update checkpoint
				if checkpoint != nil {
					mu.Lock()
					if checkpoint.ProcessedPages == nil {
						checkpoint.ProcessedPages = make(map[string][]int)
					}
					checkpoint.ProcessedPages[task.BaseName] = append(checkpoint.ProcessedPages[task.BaseName], job.PageNum)
					mu.Unlock()
				}
			}
		}()
	}

	wg.Wait()

	sort.Slice(results, func(i, j int) bool {
		return results[i].PageNum < results[j].PageNum
	})

	return results
}

func writePageOutput(task *PDFTask, pageNum int, imgPath, text string, cfg Config) error {
	var content string
	relImgPath := fmt.Sprintf("../images/%s", filepath.Base(imgPath))
	trimmedText := strings.TrimSpace(text)

	// Apply pretty formatting if enabled
	if cfg.Pretty && cfg.Format == "md" {
		trimmedText = formatText(text)
	}

	switch cfg.Format {
	case "json":
		type pageData struct {
			PageNum int    `json:"page_num"`
			Image   string `json:"image"`
			Text    string `json:"text"`
		}
		data := pageData{PageNum: pageNum, Image: relImgPath, Text: trimmedText}
		jsonBytes, err := json.Marshal(data)
		if err != nil {
			return err
		}
		content = string(jsonBytes) + "\n"
	case "txt":
		content = fmt.Sprintf("--- Page %d ---\n%s\n\n", pageNum, trimmedText)
	default: // md
		var sb strings.Builder
		if !cfg.StripHeader {
			sb.WriteString(fmt.Sprintf("# Page %d\n\n", pageNum))
			sb.WriteString(fmt.Sprintf("![Page %d](%s)\n\n", pageNum, relImgPath))
		}
		sb.WriteString(trimmedText)
		sb.WriteString("\n")
		content = sb.String()
	}

	ext := ".md"
	if cfg.Format == "json" {
		ext = ".json"
	} else if cfg.Format == "txt" {
		ext = ".txt"
	}

	mdFileName := fmt.Sprintf("page_%03d%s", pageNum, ext)
	mdPath := filepath.Join(task.PagesMDDir, mdFileName)
	return os.WriteFile(mdPath, []byte(content), 0644)
}

func runTesseract(imgPath, lang string) (string, error) {
	var lastErr error
	for attempt := 1; attempt <= 3; attempt++ {
		cmd := exec.Command("tesseract", imgPath, "stdout", "-l", lang, "--psm", "6")
		var out bytes.Buffer
		var errOut bytes.Buffer
		cmd.Stdout = &out
		cmd.Stderr = &errOut

		if err := cmd.Run(); err != nil {
			lastErr = fmt.Errorf("%v: %s", err, errOut.String())
			if attempt < 3 {
				time.Sleep(time.Duration(attempt) * 100 * time.Millisecond)
			}
			continue
		}
		return out.String(), nil
	}
	return "", lastErr
}

// formatText improves OCR output readability by joining broken lines,
// fixing bullet points, and adding proper spacing.
func formatText(text string) string {
	lines := strings.Split(text, "\n")
	var result strings.Builder
	var paragraph strings.Builder

	flushParagraph := func() {
		if paragraph.Len() > 0 {
			result.WriteString(strings.TrimSpace(paragraph.String()))
			result.WriteString("\n\n")
			paragraph.Reset()
		}
	}

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)

		// Empty line = paragraph break
		if trimmed == "" {
			flushParagraph()
			continue
		}

		// Fix bullet points: lines starting with "e " that look like bullets
		if strings.HasPrefix(trimmed, "e ") && len(trimmed) > 2 {
			rest := strings.TrimPrefix(trimmed, "e ")
			// Convert to markdown bullet if it looks like a list item
			if len(rest) > 10 && !strings.Contains(rest, ".com") && !strings.Contains(rest, "http") {
				// Check if first char is uppercase (likely a list item)
				firstChar := []rune(rest)[0]
				if unicode.IsUpper(firstChar) {
					trimmed = "- " + rest
				}
			}
		}

		// Check if line starts a new section (short line ending with :)
		isLabel := strings.HasSuffix(trimmed, ":") && len(trimmed) < 50

		if paragraph.Len() > 0 {
			// Join with space - tesseract hard-wraps lines
			paragraph.WriteString(" ")
		}
		paragraph.WriteString(trimmed)

		// Flush if line ends with sentence terminator or is a label
		if isLabel || strings.HasSuffix(trimmed, ".") || strings.HasSuffix(trimmed, ":") {
			flushParagraph()
		}
	}

	flushParagraph()
	return result.String()
}

// BatchTesseract runs OCR on multiple images in a single tesseract invocation
// This is more efficient for large batches
func batchTesseract(imgPaths []string, lang string) (map[string]string, error) {
	results := make(map[string]string, len(imgPaths))
	var mu sync.Mutex
	var wg sync.WaitGroup

	for _, path := range imgPaths {
		wg.Add(1)
		go func(p string) {
			defer wg.Done()
			text, err := runTesseract(p, lang)
			mu.Lock()
			if err != nil {
				results[p] = fmt.Sprintf("[OCR Error: %v]", err)
			} else {
				results[p] = text
			}
			mu.Unlock()
		}(path)
	}

	wg.Wait()
	return results, nil
}

func generateCombinedMarkdown(title, outPath string, pages []*PageResult, skipEmpty bool) error {
	var sb strings.Builder
	now := time.Now().Format("2006-01-02 15:04:05")

	sb.WriteString(fmt.Sprintf("# %s\n\n", title))
	sb.WriteString(fmt.Sprintf("> OCR Exported on %s | Total Pages: %d\n\n", now, len(pages)))
	sb.WriteString("---\n\n")

	for _, p := range pages {
		// Skip empty pages if flag is set
		if skipEmpty && strings.TrimSpace(p.Text) == "" {
			continue
		}
		sb.WriteString(fmt.Sprintf("## Page %d\n\n", p.PageNum))
		sb.WriteString(fmt.Sprintf("![Page %d](images/%s)\n\n", p.PageNum, p.FileName))
		if text := strings.TrimSpace(p.Text); text != "" {
			sb.WriteString(text)
			sb.WriteString("\n\n")
		}
		sb.WriteString("---\n\n")
	}

	return os.WriteFile(outPath, []byte(sb.String()), 0644)
}

func generateJSONOutput(task *PDFTask, pages []*PageResult, skipEmpty bool) error {
	type document struct {
		Title      string       `json:"title"`
		Exported   string       `json:"exported"`
		TotalPages int          `json:"total_pages"`
		Pages      []*PageResult `json:"pages"`
	}

	filtered := pages
	if skipEmpty {
		filtered = make([]*PageResult, 0, len(pages))
		for _, p := range pages {
			if strings.TrimSpace(p.Text) != "" {
				filtered = append(filtered, p)
			}
		}
	}

	doc := document{
		Title:      task.BaseName,
		Exported:   time.Now().Format("2006-01-02T15:04:05"),
		TotalPages: len(filtered),
		Pages:      filtered,
	}

	jsonPath := filepath.Join(task.MDDir, task.BaseName+".json")
	jsonBytes, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(jsonPath, jsonBytes, 0644)
}

func generateTextOutput(task *PDFTask, pages []*PageResult, skipEmpty bool) error {
	var sb strings.Builder

	filtered := pages
	if skipEmpty {
		filtered = make([]*PageResult, 0, len(pages))
		for _, p := range pages {
			if strings.TrimSpace(p.Text) != "" {
				filtered = append(filtered, p)
			}
		}
	}

	sb.WriteString(fmt.Sprintf("DOCUMENT: %s\n", task.BaseName))
	sb.WriteString(fmt.Sprintf("Exported: %s\n", time.Now().Format("2006-01-02 15:04:05")))
	sb.WriteString(fmt.Sprintf("Pages: %d\n\n", len(filtered)))
	sb.WriteString(strings.Repeat("=", 80) + "\n\n")

	for _, p := range filtered {
		sb.WriteString(fmt.Sprintf("--- Page %d ---\n\n", p.PageNum))
		sb.WriteString(strings.TrimSpace(p.Text))
		sb.WriteString("\n\n")
	}

	txtPath := filepath.Join(task.MDDir, task.BaseName+".txt")
	return os.WriteFile(txtPath, []byte(sb.String()), 0644)
}

func sortImagePaths(paths []string) {
	sort.Slice(paths, func(i, j int) bool {
		m1 := rePageNumSort.FindStringSubmatch(filepath.Base(paths[i]))
		m2 := rePageNumSort.FindStringSubmatch(filepath.Base(paths[j]))
		if len(m1) > 1 && len(m2) > 1 {
			n1, _ := strconv.Atoi(m1[1])
			n2, _ := strconv.Atoi(m2[1])
			return n1 < n2
		}
		return paths[i] < paths[j]
	})
}

func extractPageNum(path string, defaultNum int) int {
	matches := rePageNum.FindStringSubmatch(filepath.Base(path))
	if len(matches) > 1 {
		if num, err := strconv.Atoi(matches[1]); err == nil {
			return num
		}
	}
	return defaultNum
}

// Checkpoint functions
func loadCheckpoint(path string) *Checkpoint {
	data, err := os.ReadFile(path)
	if err != nil {
		// Return new checkpoint if file doesn't exist
		return &Checkpoint{
			ProcessedPDFs:  make([]string, 0),
			ProcessedPages: make(map[string][]int),
		}
	}

	var cp Checkpoint
	if err := json.Unmarshal(data, &cp); err != nil {
		return &Checkpoint{
			ProcessedPDFs:  make([]string, 0),
			ProcessedPages: make(map[string][]int),
		}
	}
	return &cp
}

func saveCheckpoint(path string, cp *Checkpoint) error {
	data, err := json.MarshalIndent(cp, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// ProgressReader wraps a reader to report progress
type ProgressReader struct {
	Reader     io.Reader
	Total      int64
	Current    int64
	OnProgress func(current, total int64)
}

func (pr *ProgressReader) Read(p []byte) (int, error) {
	n, err := pr.Reader.Read(p)
	pr.Current += int64(n)
	if pr.OnProgress != nil {
		pr.OnProgress(pr.Current, pr.Total)
	}
	return n, err
}

func plural(n int) string {
	if n == 1 {
		return ""
	}
	return "s"
}
