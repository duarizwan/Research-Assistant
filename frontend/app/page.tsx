"use client";
import React, { useState, useRef, useEffect } from "react";
import {
  Send,
  Sun,
  Moon,
  Bot,
  User,
  FileText,
  Loader2,
  ChevronRight,
  CheckCircle,
  Search,
  BookOpen,
  FolderDown,
  FileCheck,
} from "lucide-react";

interface Paper {
  id: string;
  title: string;
  authors: string[];
  published: string;
  status: string;
  summary: string;
  pdf_url?: string;
  categories?: string[];
}

interface Message {
  id: number;
  type: "user" | "bot";
  content: string;
  timestamp: Date;
  papers?: Paper[];
  workflowStep?: WorkflowStep;
}

interface MessageProps {
  message: Message;
  isDark: boolean;
}

type WorkflowStep =
  | "greeting"
  | "field_selection"
  | "topic_selection"
  | "paper_listing"
  | "load_more"
  | "download_count"
  | "paper_selection"
  | "download_confirmation"
  | "downloading"
  | "completed";

interface WorkflowState {
  step: WorkflowStep;
  field?: string;
  topic?: string;
  papers?: Paper[];
  downloadCount?: number;
  selectedPapers?: string[];
  totalPaperCount?: number;
  globalPaperMap?: { [key: number]: Paper };
}

// Helper function to parse paper selection with global serial numbers
const parseGlobalPaperSelection = (
  selection: string,
  globalPaperMap: { [key: number]: Paper } | undefined,
  maxCount: number
): Paper[] => {
  const selectedPapers: Paper[] = [];
  const parts = selection.split(",").map((part) => part.trim());

  for (const part of parts) {
    if (selectedPapers.length >= maxCount) break;

    // Check if it's a number (global serial number)
    const num = parseInt(part);
    if (!isNaN(num) && globalPaperMap && globalPaperMap[num]) {
      const paper = globalPaperMap[num];
      if (paper && !selectedPapers.find((p) => p.id === paper.id)) {
        selectedPapers.push(paper);
      }
    } else {
      // Check for author name or year across all papers
      if (globalPaperMap) {
        const allPapers = Object.values(
          globalPaperMap as { [key: number]: Paper }
        );
        const matchingPapers = allPapers.filter((paper) => {
          const authorMatch = paper.authors.some((author) =>
            author.toLowerCase().includes(part.toLowerCase())
          );
          const yearMatch = paper.published.includes(part);
          return authorMatch || yearMatch;
        });

        for (const paper of matchingPapers) {
          if (selectedPapers.length >= maxCount) break;
          if (!selectedPapers.find((p) => p.id === paper.id)) {
            selectedPapers.push(paper);
          }
        }
      }
    }
  }

  return selectedPapers;
};

// Counter for unique message IDs
let messageIdCounter = 0;

// Helper function to generate unique message IDs
const generateMessageId = (): number => {
  messageIdCounter += 1;
  return Date.now() + messageIdCounter;
};

const ResearchChatUI = () => {
  const [isDark, setIsDark] = useState(true);
  const [workflow, setWorkflow] = useState<WorkflowState>({
    step: "greeting",
  });
  const [messages, setMessages] = useState<Message[]>([]);
  const [isClient, setIsClient] = useState(false);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const inputRef = useRef<HTMLInputElement | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    setIsClient(true);
    // Initialize with welcome message only on client
    if (messages.length === 0) {
      setMessages([
        {
          id: generateMessageId(),
          type: "bot",
          content:
            "Hello 👋 I'm your Research Assistant! I can help you find and download academic papers from arXiv. Let's get started with your research journey!",
          timestamp: new Date(),
          workflowStep: "greeting",
        },
      ]);
    }
  }, [messages.length]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: generateMessageId(),
      type: "user",
      content: inputValue,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    const currentInput = inputValue;
    setInputValue("");
    setIsLoading(true);

    try {
      // Handle quit command globally
      const normalizedInput = currentInput.toLowerCase().trim();
      if (normalizedInput === "quit" || normalizedInput === "q") {
        await handleRestartOrContinue(currentInput);
      } else {
        await handleWorkflowStep(currentInput);
      }
    } catch (error) {
      console.error("Error:", error);
      const errorMessage: Message = {
        id: generateMessageId(),
        type: "bot",
        content: "Sorry, I encountered an error. Please try again.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleWorkflowStep = async (input: string) => {
    switch (workflow.step) {
      case "greeting":
        await handleFieldSelection();
        break;
      case "field_selection":
        await handleTopicSelection(input);
        break;
      case "topic_selection":
        await handlePaperSearch(input);
        break;
      case "paper_listing":
        await handleLoadMore(input);
        break;
      case "load_more":
        await handleLoadMore(input);
        break;
      case "download_count":
        await handleDownloadCount(input);
        break;
      case "paper_selection":
        await handlePaperSelection(input);
        break;
      case "download_confirmation":
        await handleDownloadConfirmation(input);
        break;
      case "downloading":
        // User can't interact during download
        break;
      case "completed":
        await handleRestartOrContinue(input);
        break;
      default:
        break;
    }
  };

  const handleFieldSelection = async () => {
    setWorkflow((prev) => ({ ...prev, step: "field_selection" }));
    const botMessage: Message = {
      id: generateMessageId(),
      type: "bot",
      content:
        "Which field of research are you looking for papers in? (e.g., Computer Science, Biology, Physics, Mathematics, Chemistry)",
      timestamp: new Date(),
      workflowStep: "field_selection",
    };
    setMessages((prev) => [...prev, botMessage]);
  };

  const handleTopicSelection = async (field: string) => {
    setWorkflow((prev) => ({ ...prev, step: "topic_selection", field }));
    const suggestions = getFieldSuggestions(field);

    let content = `Great! Now, which specific topic in ${field} are you interested in?`;
    if (suggestions.length > 0) {
      content += `\n\nSome suggestions: ${suggestions.join(", ")}`;
    }

    const botMessage: Message = {
      id: generateMessageId(),
      type: "bot",
      content: content,
      timestamp: new Date(),
      workflowStep: "topic_selection",
    };
    setMessages((prev) => [...prev, botMessage]);
  };

  const handlePaperSearch = async (topic: string) => {
    setWorkflow((prev) => ({ ...prev, step: "paper_listing", topic }));

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: `${workflow.field} ${topic}`,
          workflow_step: "search",
          max_results: 10, // Request exactly 10 papers for first batch
        }),
      });

      const data = await response.json();

      if (data.papers && data.papers.length > 0) {
        // The backend now handles sequential numbering
        const papersToShow = data.papers;

        // Create global paper map from backend response
        const globalPaperMap: { [key: number]: Paper } = {};
        papersToShow.forEach((paper: Paper, index: number) => {
          // Backend assigns sequential numbers starting from 1
          globalPaperMap[index + 1] = paper;
          console.log(
            `Initial: Adding paper ${paper.id} at global index ${index + 1}`
          );
        });

        setWorkflow((prev) => ({
          ...prev,
          papers: papersToShow,
          step: "paper_listing",
          totalPaperCount: papersToShow.length,
          globalPaperMap,
        }));

        // First message: Show papers with backend response
        const papersMessage: Message = {
          id: generateMessageId(),
          type: "bot",
          content:
            data.response ||
            `Found ${papersToShow.length} research papers on "${topic}" in ${workflow.field}!`,
          timestamp: new Date(),
          papers: papersToShow,
          workflowStep: "paper_listing",
        };
        setMessages((prev) => [...prev, papersMessage]);

        // Second message: Ask the question separately
        setTimeout(() => {
          const questionMessage: Message = {
            id: generateMessageId(),
            type: "bot",
            content:
              papersToShow.length < 10
                ? `Found ${papersToShow.length} papers. Would you like to search for more papers on this topic? (yes/no)`
                : "Would you like to search for more papers on this topic? (yes/no)",
            timestamp: new Date(),
            workflowStep: "paper_listing",
          };
          setMessages((prev) => [...prev, questionMessage]);
        }, 1000);
      } else {
        const botMessage: Message = {
          id: generateMessageId(),
          type: "bot",
          content: `Sorry, no papers found for "${topic}" in ${workflow.field}. Let's try a different topic or field.`,
          timestamp: new Date(),
          workflowStep: "field_selection",
        };
        setMessages((prev) => [...prev, botMessage]);
        setWorkflow((prev) => ({
          ...prev,
          step: "field_selection",
          field: undefined,
          topic: undefined,
        }));
      }
    } catch (error) {
      throw error;
    }
  };

  const handleLoadMore = async (input: string) => {
    const normalizedInput = input.toLowerCase().trim();

    if (normalizedInput === "yes" || normalizedInput === "y") {
      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            message: `${workflow.field} ${workflow.topic}`, // Use the original search query
            workflow_step: "load_more",
            field: workflow.field,
            topic: workflow.topic,
            max_results: 5, // Request exactly 5 papers for load more
          }),
        });

        const data = await response.json();

        if (data.papers && data.papers.length > 0) {
          // Add new papers to existing papers and update global map
          setWorkflow((prev) => {
            const newPapers = [...(prev.papers || []), ...data.papers];
            const newGlobalPaperMap = { ...prev.globalPaperMap };

            // Backend handles sequential numbering, so we need to get the correct numbers
            // The backend response should include the correct sequential numbers
            data.papers.forEach((paper: Paper, index: number) => {
              // Calculate the sequential number based on current count
              const globalIndex = (prev.totalPaperCount || 0) + index + 1;
              newGlobalPaperMap[globalIndex] = paper;
              console.log(
                `Adding paper ${paper.id} at global index ${globalIndex}`
              );
            });

            return {
              ...prev,
              papers: newPapers,
              totalPaperCount: (prev.totalPaperCount || 0) + data.papers.length,
              globalPaperMap: newGlobalPaperMap,
              step: "paper_listing", // Keep on papers step
            };
          });

          // First message: Show new papers with backend response
          const papersMessage: Message = {
            id: generateMessageId(),
            type: "bot",
            content:
              data.response ||
              `Found ${data.papers.length} additional papers on the same topic!`,
            timestamp: new Date(),
            papers: data.papers,
            workflowStep: "paper_listing",
          };
          setMessages((prev) => [...prev, papersMessage]);

          // Second message: Ask the question separately
          setTimeout(() => {
            const questionMessage: Message = {
              id: generateMessageId(),
              type: "bot",
              content:
                "Would you like to search for more papers on this topic? (yes/no)",
              timestamp: new Date(),
              workflowStep: "paper_listing",
            };
            setMessages((prev) => [...prev, questionMessage]);
          }, 1000);
        } else {
          // No more papers available - check if it's because no papers found or user said no
          if (
            data.response &&
            data.response.includes("No more relevant papers available")
          ) {
            // No more papers available on arXiv
            const botMessage: Message = {
              id: generateMessageId(),
              type: "bot",
              content: data.response,
              timestamp: new Date(),
              workflowStep: "download_count",
            };
            setMessages((prev) => [...prev, botMessage]);
            setWorkflow((prev) => ({ ...prev, step: "download_count" }));
          } else {
            // Other error case
            const botMessage: Message = {
              id: generateMessageId(),
              type: "bot",
              content:
                "No more papers available. How many papers would you like to download?",
              timestamp: new Date(),
              workflowStep: "download_count",
            };
            setMessages((prev) => [...prev, botMessage]);
            setWorkflow((prev) => ({ ...prev, step: "download_count" }));
          }
        }
      } catch (error) {
        console.error("Error loading more papers:", error);
        const errorMessage: Message = {
          id: generateMessageId(),
          type: "bot",
          content: "Error loading more papers. Please try again.",
          timestamp: new Date(),
          workflowStep: "paper_listing",
        };
        setMessages((prev) => [...prev, errorMessage]);
      }
    } else if (normalizedInput === "no" || normalizedInput === "n") {
      setWorkflow((prev) => ({ ...prev, step: "download_count" }));
      const botMessage: Message = {
        id: generateMessageId(),
        type: "bot",
        content:
          "Continuing with current papers. How many papers would you like to download?",
        timestamp: new Date(),
        workflowStep: "download_count",
      };
      setMessages((prev) => [...prev, botMessage]);
    } else {
      const errorMessage: Message = {
        id: generateMessageId(),
        type: "bot",
        content:
          "Invalid command. Please enter 'yes' to load more papers or 'no' to continue.",
        timestamp: new Date(),
        workflowStep: "paper_listing",
      };
      setMessages((prev) => [...prev, errorMessage]);
    }
  };

  const handleDownloadCount = async (countStr: string) => {
    console.log("Processing download count:", countStr);
    console.log("Current workflow state:", workflow);

    // Handle cases where user might enter multiple numbers or commas
    let cleanedInput = countStr.trim();

    // If user entered something like "8,6" or "8 6", take the first number
    if (cleanedInput.includes(",") || cleanedInput.includes(" ")) {
      const numbers = cleanedInput.split(/[,\s]+/);
      cleanedInput = numbers[0];
      console.log("Found multiple values, using first:", cleanedInput);
    }

    const count = parseInt(cleanedInput);
    const maxPapers = workflow.totalPaperCount || 0;

    if (isNaN(count) || count < 1 || count > maxPapers) {
      const botMessage: Message = {
        id: generateMessageId(),
        type: "bot",
        content: `Please enter a valid number between 1 and ${maxPapers}. ${
          countStr.includes(",")
            ? "I noticed you entered multiple numbers - please enter just one number for how many papers to download."
            : ""
        }`,
        timestamp: new Date(),
        workflowStep: "download_count",
      };
      setMessages((prev) => [...prev, botMessage]);
      return;
    }

    console.log("Setting download count to:", count);
    setWorkflow((prev) => ({
      ...prev,
      step: "paper_selection",
      downloadCount: count,
    }));

    const botMessage: Message = {
      id: generateMessageId(),
      type: "bot",
      content: `👉 Selection options:\n   - Serial numbers: 1,3,5\n   - Author name: Goodfellow\n   - Year: 2020\n   - Mixed: 1,Goodfellow,2020\n   - 'help' for more guidance\n\nEnter your selection:`,
      timestamp: new Date(),
      workflowStep: "paper_selection",
    };
    setMessages((prev) => [...prev, botMessage]);
  };

  const handlePaperSelection = async (selection: string) => {
    if (!workflow.papers || !workflow.downloadCount) {
      const errorMessage: Message = {
        id: generateMessageId(),
        type: "bot",
        content:
          "❌ Error: No papers or download count available. Please search for papers again.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
      return;
    }

    if (selection.toLowerCase().trim() === "help") {
      const helpMessage: Message = {
        id: generateMessageId(),
        type: "bot",
        content: `💡 Selection Help:\n   - Use paper numbers from the list above\n   - Type author's last name (e.g., 'Smith' matches 'John Smith')\n   - Use 4-digit years (e.g., '2020', '2023')\n   - Combine methods with commas: '1,2,Smith,2020'\n\nEnter your selection:`,
        timestamp: new Date(),
        workflowStep: "paper_selection",
      };
      setMessages((prev) => [...prev, helpMessage]);
      return;
    }

    // Parse the selection using global paper map
    const selectedPapers = parseGlobalPaperSelection(
      selection,
      workflow.globalPaperMap,
      workflow.downloadCount || 0
    );

    if (selectedPapers.length === 0) {
      const errorMessage: Message = {
        id: generateMessageId(),
        type: "bot",
        content:
          "⚠️ No matching papers found for your selection. Please try again with different criteria.",
        timestamp: new Date(),
        workflowStep: "paper_selection",
      };
      setMessages((prev) => [...prev, errorMessage]);
      return;
    }

    if (selectedPapers.length < workflow.downloadCount) {
      const confirmMessage: Message = {
        id: generateMessageId(),
        type: "bot",
        content: `⚠️ You requested ${workflow.downloadCount} papers but selected ${selectedPapers.length}. Continue with ${selectedPapers.length} papers? (y/n)`,
        timestamp: new Date(),
        workflowStep: "download_confirmation",
      };
      setMessages((prev) => [...prev, confirmMessage]);
      // Store the selection temporarily (store paper IDs, not Paper objects)
      setWorkflow((prev) => ({
        ...prev,
        selectedPapers: selectedPapers.map((p) => p.id),
        step: "download_confirmation",
      }));
      return;
    }

    // Handle y/n responses for confirmation
    if (selection.toLowerCase().trim() === "y" && workflow.selectedPapers) {
      const finalSelectionMessage: Message = {
        id: generateMessageId(),
        type: "bot",
        content: `📋 Final selection (${
          workflow.selectedPapers.length
        } papers):\n${workflow.selectedPapers
          .map((paperId, index) => {
            const paper = workflow.papers?.find((p) => p.id === paperId);
            return `${index + 1}) ${
              paper?.title.substring(0, 65) || "Unknown"
            }... (${paperId})`;
          })
          .join("\n")}\n\n✅ Proceed with download? (y/n/preview)`,
        timestamp: new Date(),
        workflowStep: "downloading",
      };
      setMessages((prev) => [...prev, finalSelectionMessage]);
      setWorkflow((prev) => ({ ...prev, step: "downloading" }));
      await handleDownloadStart(
        workflow.selectedPapers
          .map((id) => workflow.papers?.find((p) => p.id === id))
          .filter(Boolean) as Paper[]
      );
      return;
    }

    if (selection.toLowerCase().trim() === "n") {
      const retryMessage: Message = {
        id: generateMessageId(),
        type: "bot",
        content: `👉 Selection options:\n   - Serial numbers: 1,3,5\n   - Author name: Goodfellow\n   - Year: 2020\n   - Mixed: 1,Goodfellow,2020\n   - 'help' for more guidance\n\nEnter your selection:`,
        timestamp: new Date(),
        workflowStep: "paper_selection",
      };
      setMessages((prev) => [...prev, retryMessage]);
      setWorkflow((prev) => ({ ...prev, selectedPapers: undefined }));
      return;
    }

    // Handle y/n responses for final download confirmation
    if (selection.toLowerCase().trim() === "y") {
      // Start the download process
      await handleDownloadStart(selectedPapers);
      return;
    }

    if (selection.toLowerCase().trim() === "n") {
      const cancelMessage: Message = {
        id: generateMessageId(),
        type: "bot",
        content:
          "❌ Download cancelled. Would you like to search for more papers? Just say 'yes' to start a new search!",
        timestamp: new Date(),
        workflowStep: "completed",
      };
      setMessages((prev) => [...prev, cancelMessage]);
      setWorkflow((prev) => ({ ...prev, step: "completed" }));
      return;
    }

    if (selection.toLowerCase().trim() === "preview") {
      const previewMessage: Message = {
        id: generateMessageId(),
        type: "bot",
        content: `📋 Preview of ${
          selectedPapers.length
        } papers to be downloaded:\n${selectedPapers
          .map(
            (paper, index) =>
              `${index + 1}) ${paper.title.substring(0, 65)}... (${paper.id})`
          )
          .join("\n")}\n\n✅ Proceed with download? (y/n)`,
        timestamp: new Date(),
        workflowStep: "downloading",
      };
      setMessages((prev) => [...prev, previewMessage]);
      setWorkflow((prev) => ({
        ...prev,
        step: "downloading",
        selectedPapers: selectedPapers.map((p) => p.id),
      }));
      return;
    }

    // Show final selection
    const finalSelectionMessage: Message = {
      id: generateMessageId(),
      type: "bot",
      content: `📋 Final selection (${
        selectedPapers.length
      } papers):\n${selectedPapers
        .map(
          (paper, index) =>
            `${index + 1}) ${paper.title.substring(0, 65)}... (${paper.id})`
        )
        .join("\n")}\n\n✅ Proceed with download? (y/n/preview)`,
      timestamp: new Date(),
      workflowStep: "download_confirmation",
      papers: selectedPapers,
    };
    setMessages((prev) => [...prev, finalSelectionMessage]);
    setWorkflow((prev) => ({
      ...prev,
      step: "download_confirmation",
      selectedPapers: selectedPapers.map((p) => p.id),
    }));
  };

  const handleDownloadConfirmation = async (input: string) => {
    const normalizedInput = input.toLowerCase().trim();

    if (normalizedInput === "y" || normalizedInput === "yes") {
      // Check if this is a count mismatch confirmation
      if (
        workflow.selectedPapers &&
        workflow.downloadCount &&
        workflow.selectedPapers.length < workflow.downloadCount
      ) {
        // User confirmed to continue with fewer papers than requested
        const finalSelectionMessage: Message = {
          id: generateMessageId(),
          type: "bot",
          content: `📋 Final selection (${
            workflow.selectedPapers.length
          } papers):\n${workflow.selectedPapers
            .map((paperId, index) => {
              const paper = workflow.papers?.find((p) => p.id === paperId);
              return `${index + 1}) ${
                paper?.title.substring(0, 65) || "Unknown"
              }... (${paperId})`;
            })
            .join("\n")}\n\n✅ Proceed with download? (y/n/preview)`,
          timestamp: new Date(),
          workflowStep: "download_confirmation",
          papers: workflow.selectedPapers
            .map((id) => workflow.papers?.find((p) => p.id === id))
            .filter(Boolean) as Paper[],
        };
        setMessages((prev) => [...prev, finalSelectionMessage]);
        setWorkflow((prev) => ({ ...prev, step: "download_confirmation" }));
        return;
      } else {
        // Show download button interface
        const downloadInterfaceMessage: Message = {
          id: generateMessageId(),
          type: "bot",
          content: `📋 Final selection (${
            workflow.selectedPapers?.length || 0
          } papers):\n${workflow.selectedPapers
            ?.map((paperId, index) => {
              const paper = workflow.papers?.find((p) => p.id === paperId);
              return `${index + 1}) ${
                paper?.title.substring(0, 65) || "Unknown"
              }... (${paperId})`;
            })
            .join(
              "\n"
            )}\n\n✅ Ready to download! Click the button below to download all papers:`,
          timestamp: new Date(),
          workflowStep: "download_confirmation",
          papers: workflow.selectedPapers
            ?.map((id) => workflow.papers?.find((p) => p.id === id))
            .filter(Boolean) as Paper[],
        };
        setMessages((prev) => [...prev, downloadInterfaceMessage]);
        setWorkflow((prev) => ({ ...prev, step: "download_confirmation" }));
        return;
      }
    }

    if (normalizedInput === "n" || normalizedInput === "no") {
      const retryMessage: Message = {
        id: generateMessageId(),
        type: "bot",
        content: `👉 Selection options:\n   - Serial numbers: 1,3,5\n   - Author name: Goodfellow\n   - Year: 2020\n   - Mixed: 1,Goodfellow,2020\n   - 'help' for more guidance\n\nEnter your selection:`,
        timestamp: new Date(),
        workflowStep: "paper_selection",
      };
      setMessages((prev) => [...prev, retryMessage]);
      setWorkflow((prev) => ({
        ...prev,
        selectedPapers: undefined,
        step: "paper_selection",
      }));
      return;
    }

    if (normalizedInput === "preview") {
      const selectedPapers = workflow.selectedPapers
        ?.map((id) => workflow.papers?.find((p) => p.id === id))
        .filter(Boolean) as Paper[];

      const previewMessage: Message = {
        id: generateMessageId(),
        type: "bot",
        content: `📋 Preview of ${
          selectedPapers.length
        } papers to be downloaded:\n${selectedPapers
          .map(
            (paper, index) =>
              `${index + 1}) ${paper.title.substring(0, 65)}... (${paper.id})`
          )
          .join("\n")}\n\n✅ Proceed with download? (y/n)`,
        timestamp: new Date(),
        workflowStep: "download_confirmation",
      };
      setMessages((prev) => [...prev, previewMessage]);
      return;
    }

    // Invalid input
    const errorMessage: Message = {
      id: generateMessageId(),
      type: "bot",
      content:
        "Invalid command. Please enter 'y' to proceed, 'n' to reselect, or 'preview' to see details.",
      timestamp: new Date(),
      workflowStep: "download_confirmation",
    };
    setMessages((prev) => [...prev, errorMessage]);
  };

  const handleDownloadButtonClick = async () => {
    const selectedPapers = workflow.selectedPapers
      ?.map((id) => workflow.papers?.find((p) => p.id === id))
      .filter(Boolean) as Paper[];

    if (selectedPapers && selectedPapers.length > 0) {
      // Open file browser for user to select save location
      const input = document.createElement("input");
      input.type = "file";
      input.webkitdirectory = true;
      input.style.display = "none";
      document.body.appendChild(input);

      input.addEventListener("change", async (event) => {
        const target = event.target as HTMLInputElement;
        if (target.files && target.files.length > 0) {
          const selectedFolder =
            target.files[0].webkitRelativePath.split("/")[0];
          console.log(`Selected folder: ${selectedFolder}`);

          // Download each PDF to the selected folder
          for (const paper of selectedPapers) {
            if (paper.pdf_url) {
              try {
                // Fetch the PDF content
                const response = await fetch(paper.pdf_url);
                const blob = await response.blob();

                // Create safe filename
                const safeTitle = paper.title
                  .substring(0, 40)
                  .replace(/[^a-zA-Z0-9]/g, "_");
                const filename = `${paper.id}_${safeTitle}.pdf`;

                // Create download link with the selected folder path
                const link = document.createElement("a");
                link.href = URL.createObjectURL(blob);
                link.download = filename;
                link.target = "_blank";
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                URL.revokeObjectURL(link.href);

                console.log(`Downloaded ${filename} to ${selectedFolder}`);
              } catch (error) {
                console.error(`Failed to download ${paper.title}:`, error);
              }
            }
          }
        }
        document.body.removeChild(input);
      });

      input.click();

      // Then start the backend download process
      setWorkflow((prev) => ({ ...prev, step: "downloading" }));
      await handleDownloadStart(selectedPapers);
    }
  };

  const handleDownloadStart = async (selectedPapers: Paper[]) => {
    console.log("handleDownloadStart called with papers:", selectedPapers);
    console.log("Current workflow state at download start:", workflow);

    // Validation checks
    if (!selectedPapers || selectedPapers.length === 0) {
      console.error("Download attempted without selected papers");
      return;
    }

    if (!workflow.field || !workflow.topic) {
      const errorMessage: Message = {
        id: generateMessageId(),
        type: "bot",
        content:
          "❌   Missing Information  \n\nField and topic information are required for download. Please search for papers again.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
      return;
    }

    try {
      // Start the actual download process
      const downloadResponse = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: "download papers",
          workflow_step: "download",
          paper_ids: selectedPapers.map((p) => p.id),
          field: workflow.field,
          topic: workflow.topic,
        }),
      });

      if (!downloadResponse.ok) {
        let errorMessage = "Download request failed";
        try {
          const errorData = await downloadResponse.json();
          errorMessage =
            errorData.message || errorData.response || errorMessage;
        } catch {
          // Use default error message if JSON parsing fails
        }
        throw new Error(
          `Server error (${downloadResponse.status}): ${errorMessage}`
        );
      }

      // Parse the response to check for backend errors
      const downloadResult = await downloadResponse.json();
      if (downloadResult.action_type === "error") {
        throw new Error(downloadResult.response || "Download failed on server");
      }

      // Get papers with detailed summaries from backend
      const papersWithSummaries = downloadResult.papers || selectedPapers;

      // Show real-time feedback with actual paper details and detailed summaries
      for (let i = 0; i < papersWithSummaries.length; i++) {
        const paper = papersWithSummaries[i];

        // Extract year from published date
        const year = paper.published
          ? paper.published.split("-")[0]
          : "Unknown";

        const progressMessage: Message = {
          id: generateMessageId(),
          type: "bot",
          content: `📥 Downloading ${i + 1}/${
            papersWithSummaries.length
          }:\n\n📄   ${paper.title}  \n👥   Authors:   ${paper.authors
            .slice(0, 3)
            .join(", ")}${
            paper.authors.length > 3 ? " et al." : ""
          }\n📅 Year:   ${year}\n📋   Status:   ${
            paper.status
          }\n\n📝   Summary (100-150 words):\n\n${paper.summary}`,
          timestamp: new Date(),
          workflowStep: "downloading",
        };
        setMessages((prev) => [...prev, progressMessage]);

        // Realistic download delay (papers are actually downloading in background)
        await new Promise((resolve) => setTimeout(resolve, 3000));

        // Show completion for individual paper
        const completedMessage: Message = {
          id: generateMessageId(),
          type: "bot",
          content: `✅   Paper ${i + 1} completed:   ${paper.title.substring(
            0,
            50
          )}${paper.title.length > 50 ? "..." : ""}`,
          timestamp: new Date(),
          workflowStep: "downloading",
        };
        setMessages((prev) => [...prev, completedMessage]);
      }

      // Final completion message
      // First message: Download completion details
      const completionMessage: Message = {
        id: generateMessageId(),
        type: "bot",
        content: `🎉   Download Complete!  \n\n✅   Successfully downloaded ${
          papersWithSummaries.length
        } papers  \n\n📊   Download Summary:  \n• ✅ Successful: ${
          papersWithSummaries.length
        }\n• ⏭️ Skipped: 0\n• ❌ Failed: 0\n\n📁   Location:   papers/${
          workflow.field
        }/${workflow.topic}/\n\n📄   Files Created:  \n${papersWithSummaries
          .map(
            (paper: Paper) =>
              `• ${paper.id}_${paper.title
                .substring(0, 40)
                .replace(/[^a-zA-Z0-9]/g, "_")}.pdf`
          )
          .join(
            "\n"
          )}\n• download_summary.txt (detailed paper information)\n\n🤖   AI Summaries:   Each paper now has a concise 100-150 word AI-generated summary in the download_summary.txt file!\n\n💡   Tip:   Check your project's papers folder for all downloaded files!\n\n🙏   Thank you for using the Research Assistant!`,
        timestamp: new Date(),
        workflowStep: "completed",
      };
      setMessages((prev) => [...prev, completionMessage]);

      // Second message: Continue question (sent instantly after)
      setTimeout(() => {
        const continueMessage: Message = {
          id: generateMessageId(),
          type: "bot",
          content: `Would you like to continue searching for more papers or quit?  \n• Type   "yes"   or   "continue"   to search for more papers  \n• Type   "quit"   or   "exit"   to end the session`,
          timestamp: new Date(),
          workflowStep: "completed",
        };
        setMessages((prev) => [...prev, continueMessage]);
      }, 100);

      setWorkflow((prev) => ({ ...prev, step: "completed" }));
    } catch (error) {
      console.error("Download error:", error);

      // Create a more specific error message
      let errorContent = "❌   Download Failed  \n\n";

      if (error instanceof Error) {
        if (error.message.includes("Server error")) {
          errorContent += `🔧   Server Issue:   ${error.message}\n\n`;
          errorContent +=
            "The server encountered an issue while processing your download request. This could be due to:\n";
          errorContent += "• Temporary server overload\n";
          errorContent += "• Network connectivity issues\n";
          errorContent += "• Invalid paper selection\n\n";
        } else if (error.message.includes("Download failed on server")) {
          errorContent += `🚫   Backend Error:   ${error.message}\n\n`;
          errorContent += "Please check:\n";
          errorContent += "• That papers were properly selected\n";
          errorContent += "• Your internet connection\n";
          errorContent += "• Try searching for papers again\n\n";
        } else {
          errorContent += `📡   Network Error:   ${error.message}\n\n`;
          errorContent +=
            "Please check your internet connection and try again.\n\n";
        }
      } else {
        errorContent +=
          "An unexpected error occurred during the download process.\n\n";
      }

      errorContent += "💡   Next Steps:  \n";
      errorContent += "• Try downloading again\n";
      errorContent += "• Search for papers again if the issue persists\n";
      errorContent +=
        "• Check your project's papers folder to see if any papers were downloaded";

      const errorMessage: Message = {
        id: generateMessageId(),
        type: "bot",
        content: errorContent,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);

      // Reset workflow to allow retry
      setWorkflow((prev) => ({ ...prev, step: "paper_selection" }));
    }
  };

  const getFieldSuggestions = (field: string): string[] => {
    const suggestions: { [key: string]: string[] } = {
      "computer science": [
        "machine learning",
        "artificial intelligence",
        "computer vision",
        "natural language processing",
      ],
      physics: [
        "quantum mechanics",
        "particle physics",
        "astrophysics",
        "condensed matter",
      ],
      biology: ["genetics", "molecular biology", "neuroscience", "evolution"],
      mathematics: ["algebra", "statistics", "optimization", "topology"],
      chemistry: [
        "organic chemistry",
        "materials science",
        "biochemistry",
        "catalysis",
      ],
    };

    const fieldLower = field.toLowerCase();
    for (const [key, topics] of Object.entries(suggestions)) {
      if (fieldLower.includes(key) || key.includes(fieldLower)) {
        return topics;
      }
    }

    // No suggestions for other fields
    return [];
  };

  const restartWorkflow = () => {
    setWorkflow({ step: "greeting" });
    setMessages([
      {
        id: generateMessageId(),
        type: "bot",
        content:
          "Hello 👋 I'm your Research Assistant! I can help you find and download academic papers from arXiv. Let's get started with your research journey!",
        timestamp: new Date(),
        workflowStep: "greeting",
      },
    ]);
  };

  const getInputPlaceholder = (): string => {
    switch (workflow.step) {
      case "greeting":
        return "Type anything to start your research journey...";
      case "field_selection":
        return "Enter research field (e.g., Computer Science, Physics)...";
      case "topic_selection":
        return "Enter specific topic...";
      case "load_more":
        return "Enter 'yes' to load more papers or 'no' to continue...";
      case "download_count":
        return "Enter number of papers to download...";
      case "paper_selection":
        return "Enter selection (e.g., 1,3,5 or Smith,2020)...";
      case "download_confirmation":
        return "Enter 'y' to proceed, 'n' to reselect, or 'preview'...";
      case "completed":
        return "Type 'yes' to start a new search or ask me anything...";
      default:
        return "Type your message...";
    }
  };

  const handleRestartOrContinue = async (input: string) => {
    const normalizedInput = input.toLowerCase().trim();

    // Handle quit/q command from any step
    if (normalizedInput === "quit" || normalizedInput === "q") {
      restartWorkflow();
      return;
    }

    if (workflow.step === "completed") {
      if (
        normalizedInput === "yes" ||
        normalizedInput === "continue" ||
        normalizedInput.includes("search") ||
        normalizedInput.includes("more")
      ) {
        restartWorkflow();
        return;
      } else if (normalizedInput === "exit" || normalizedInput === "no") {
        const goodbyeMessage: Message = {
          id: generateMessageId(),
          type: "bot",
          content:
            "👋 Thank you for using the Research Assistant! Have a great day!",
          timestamp: new Date(),
          workflowStep: "completed",
        };
        setMessages((prev) => [...prev, goodbyeMessage]);
        return;
      }
    }
  };

  interface PaperCardProps {
    paper: Paper;
    isDark: boolean;
    index: number;
  }

  const PaperCard: React.FC<PaperCardProps> = ({ paper, isDark, index }) => (
    <div
      className={`p-4 rounded-lg border ${
        isDark
          ? "bg-gray-800/50 border-gray-700 hover:bg-gray-800/70"
          : "bg-gray-50 border-gray-200 hover:bg-gray-100"
      } transition-all duration-200 hover:shadow-lg`}
    >
      <div className="flex items-start justify-between mb-2">
        <span
          className={`text-xs px-2 py-1 rounded-full font-medium ${
            isDark
              ? "bg-blue-500/20 text-blue-300"
              : "bg-blue-100 text-blue-600"
          }`}
        >
          #{index + 1}
        </span>
        <div className="flex items-center gap-1">
          {paper.categories?.slice(0, 2).map((cat, i) => (
            <span
              key={i}
              className={`text-xs px-2 py-1 rounded-full ${
                isDark
                  ? "bg-purple-500/20 text-purple-300"
                  : "bg-purple-100 text-purple-600"
              }`}
            >
              {cat}
            </span>
          ))}
        </div>
      </div>

      <h4
        className={`font-semibold mb-2 line-clamp-2 ${
          isDark ? "text-blue-300" : "text-blue-600"
        }`}
      >
        {paper.title}
      </h4>

      <p
        className={`text-sm mb-2 ${isDark ? "text-gray-300" : "text-gray-600"}`}
      >
        <strong>Authors:</strong> {paper.authors.slice(0, 3).join(", ")}
        {paper.authors.length > 3 && ` et al. (${paper.authors.length} total)`}
      </p>

      <p
        className={`text-sm mb-3 ${isDark ? "text-gray-300" : "text-gray-600"}`}
      >
        <strong>Published:</strong> {paper.published} | <strong>Status:</strong>{" "}
        {paper.status}
      </p>

      {paper.summary && (
        <p
          className={`text-xs mb-3 ${
            isDark ? "text-gray-400" : "text-gray-500"
          } line-clamp-2`}
        >
          {paper.summary.substring(0, 120)}...
        </p>
      )}

      <div className="flex items-center gap-2">
        <button
          className={`px-3 py-1 text-xs rounded-full transition-all duration-200 ${
            isDark
              ? "bg-gray-700 text-gray-300 hover:bg-gray-600"
              : "bg-gray-100 text-gray-600 hover:bg-gray-200"
          }`}
          onClick={() => window.open(paper.pdf_url, "_blank")}
        >
          <FileText className="w-3 h-3 inline mr-1" />
          View PDF
        </button>
      </div>
    </div>
  );

  const WorkflowProgress: React.FC<{
    currentStep: WorkflowStep;
    isDark: boolean;
  }> = ({ currentStep, isDark }) => {
    const steps = [
      { key: "greeting", label: "Welcome", icon: Bot },
      { key: "field_selection", label: "Field", icon: BookOpen },
      { key: "topic_selection", label: "Topic", icon: Search },
      { key: "paper_listing", label: "Papers", icon: FileText },
      { key: "download_count", label: "Count", icon: ChevronRight },
      { key: "paper_selection", label: "Select", icon: FileCheck },
      { key: "downloading", label: "Download", icon: FolderDown },
      { key: "completed", label: "Done", icon: CheckCircle },
    ];

    return (
      <div
        className={`p-4 mb-4 rounded-lg border ${
          isDark
            ? "bg-gray-800/30 border-gray-700"
            : "bg-gray-50 border-gray-200"
        }`}
      >
        <h3
          className={`text-sm font-medium mb-3 ${
            isDark ? "text-gray-300" : "text-gray-600"
          }`}
        >
          Research Progress
        </h3>
        <div className="flex items-center justify-between">
          {steps.map((step, index) => {
            const Icon = step.icon;
            const isActive = step.key === currentStep;
            const isCompleted =
              steps.findIndex((s) => s.key === currentStep) > index;

            return (
              <div key={step.key} className="flex flex-col items-center">
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center transition-all duration-300 ${
                    isActive
                      ? isDark
                        ? "bg-blue-500/30 text-blue-300 shadow-lg shadow-blue-500/20"
                        : "bg-blue-500 text-white shadow-lg shadow-blue-300/30"
                      : isCompleted
                      ? isDark
                        ? "bg-green-500/30 text-green-300"
                        : "bg-green-500 text-white"
                      : isDark
                      ? "bg-gray-700 text-gray-400"
                      : "bg-gray-200 text-gray-500"
                  }`}
                >
                  <Icon className="w-4 h-4" />
                </div>
                <span
                  className={`text-xs mt-1 ${
                    isActive
                      ? isDark
                        ? "text-blue-300"
                        : "text-blue-600"
                      : isCompleted
                      ? isDark
                        ? "text-green-300"
                        : "text-green-600"
                      : isDark
                      ? "text-gray-400"
                      : "text-gray-500"
                  }`}
                >
                  {step.label}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  const MessageComponent: React.FC<MessageProps> = ({ message, isDark }) => (
    <div
      className={`flex gap-3 ${
        message.type === "user" ? "flex-row-reverse" : "flex-row"
      }`}
    >
      <div
        className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center transition-all duration-300 ${
          message.type === "user"
            ? isDark
              ? "bg-blue-500/20 text-blue-300 shadow-lg shadow-blue-500/20"
              : "bg-blue-100 text-blue-600 shadow-lg shadow-blue-200/50"
            : isDark
            ? "bg-purple-500/20 text-purple-300 shadow-lg shadow-purple-500/20"
            : "bg-purple-100 text-purple-600 shadow-lg shadow-purple-200/50"
        }`}
      >
        {message.type === "user" ? (
          <User className="w-4 h-4" />
        ) : (
          <Bot className="w-4 h-4" />
        )}
      </div>

      <div
        className={`flex-1 max-w-3xl ${
          message.type === "user" ? "text-right" : "text-left"
        }`}
      >
        <div
          className={`inline-block p-4 rounded-2xl transition-all duration-300 hover:shadow-lg ${
            message.type === "user"
              ? isDark
                ? "bg-blue-500/20 text-blue-100 shadow-lg shadow-blue-500/10 hover:shadow-blue-500/20"
                : "bg-blue-500 text-white shadow-lg shadow-blue-200/50 hover:shadow-blue-300/50"
              : isDark
              ? "bg-gray-800/60 text-gray-100 shadow-lg shadow-gray-900/20 hover:shadow-gray-900/30"
              : "bg-white text-gray-800 shadow-lg shadow-gray-200/50 hover:shadow-gray-300/50"
          }`}
        >
          <div className="whitespace-pre-wrap">{message.content}</div>

          {message.papers && (
            <div className="mt-4 space-y-3">
              <div
                className={`text-sm font-semibold ${
                  isDark ? "text-purple-300" : "text-purple-600"
                }`}
              >
                {message.workflowStep === "download_confirmation"
                  ? "Selected Papers:"
                  : "Found Papers:"}
              </div>
              {message.papers.map((paper, index) => {
                // Find the global serial number from the global paper map
                let globalIndex = index;
                if (workflow.globalPaperMap) {
                  // Find the paper in the global map to get its correct sequential number
                  const globalNumber = Object.keys(
                    workflow.globalPaperMap
                  ).find(
                    (key) =>
                      workflow.globalPaperMap![parseInt(key)].id === paper.id
                  );
                  if (globalNumber) {
                    globalIndex = parseInt(globalNumber) - 1; // Convert to 0-based index for display
                    console.log(
                      `Display: Paper ${paper.id} showing as #${
                        globalIndex + 1
                      }`
                    );
                  }
                }
                return (
                  <PaperCard
                    key={paper.id || index}
                    paper={paper}
                    isDark={isDark}
                    index={globalIndex}
                  />
                );
              })}
              {message.workflowStep === "download_confirmation" &&
                message.content.includes("Ready to download!") && (
                  <div className="mt-4 p-4 rounded-lg border border-dashed border-gray-400">
                    <div className="text-center">
                      <h3
                        className={`font-semibold mb-2 ${
                          isDark ? "text-gray-200" : "text-gray-800"
                        }`}
                      >
                        Ready to Download
                      </h3>
                      <p
                        className={`text-sm mb-2 ${
                          isDark ? "text-gray-400" : "text-gray-600"
                        }`}
                      >
                        {message.papers.length} papers selected for download
                      </p>
                      <p
                        className={`text-xs mb-4 ${
                          isDark ? "text-gray-500" : "text-gray-500"
                        }`}
                      >
                        Click to open file browser, select save location, and
                        download PDFs to your PC
                      </p>
                      <button
                        onClick={handleDownloadButtonClick}
                        className={`px-6 py-3 rounded-lg font-medium transition-all duration-300 hover:scale-105 ${
                          isDark
                            ? "bg-green-500 text-white hover:bg-green-600 shadow-lg shadow-green-500/30"
                            : "bg-green-500 text-white hover:bg-green-600 shadow-lg shadow-green-300/30"
                        }`}
                      >
                        <FolderDown className="w-5 h-5 inline mr-2" />
                        Download All Papers
                      </button>
                    </div>
                  </div>
                )}
            </div>
          )}
        </div>

        <div
          className={`text-xs mt-2 ${
            isDark ? "text-gray-400" : "text-gray-500"
          }`}
        >
          {message.timestamp.toLocaleTimeString()}
        </div>
      </div>
    </div>
  );

  // Don't render until client-side hydration is complete
  if (!isClient) {
    return (
      <div className="min-h-screen bg-black text-white flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-400">Loading Research Assistant...</p>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`min-h-screen transition-all duration-500 ${
        isDark
          ? "bg-black text-white"
          : "bg-gradient-to-br from-gray-50 via-white to-gray-50 text-gray-800"
      }`}
    >
      <div
        className={`sticky top-0 z-50 backdrop-blur-xl border-b transition-all duration-300 ${
          isDark
            ? "bg-gray-900/80 border-gray-700 shadow-xl shadow-gray-900/20"
            : "bg-white/80 border-gray-200 shadow-xl shadow-gray-200/20"
        }`}
      >
        <div className="max-w-4xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div
                className={`w-10 h-10 rounded-lg flex items-center justify-center transition-all duration-300 ${
                  isDark
                    ? "bg-gradient-to-br from-blue-500 to-purple-500 shadow-lg shadow-blue-500/25"
                    : "bg-gradient-to-br from-blue-600 to-purple-600 shadow-lg shadow-blue-300/25"
                }`}
              >
                <Bot className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1
                  className={`text-xl font-bold transition-all duration-300 ${
                    isDark
                      ? "bg-gradient-to-r from-blue-300 to-purple-300 bg-clip-text text-transparent"
                      : "bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent"
                  }`}
                >
                  Research Assistant
                </h1>
                <p
                  className={`text-xs ${
                    isDark ? "text-gray-400" : "text-gray-500"
                  }`}
                >
                  Powered by arXiv & Gemini AI
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              {workflow.step !== "greeting" && (
                <button
                  onClick={restartWorkflow}
                  className={`p-2 rounded-lg transition-all duration-300 hover:scale-110 ${
                    isDark
                      ? "bg-gray-800 hover:bg-gray-700 text-green-300 shadow-lg shadow-green-500/20 hover:shadow-green-500/30"
                      : "bg-gray-100 hover:bg-gray-200 text-green-600 shadow-lg shadow-green-300/20 hover:shadow-green-300/30"
                  }`}
                  title="Start New Search"
                >
                  <Search className="w-5 h-5" />
                </button>
              )}

              <button
                onClick={() => setIsDark(!isDark)}
                className={`p-2 rounded-lg transition-all duration-300 hover:scale-110 ${
                  isDark
                    ? "bg-gray-800 hover:bg-gray-700 text-yellow-300 shadow-lg shadow-yellow-500/20 hover:shadow-yellow-500/30"
                    : "bg-gray-100 hover:bg-gray-200 text-gray-700 shadow-lg shadow-gray-300/20 hover:shadow-gray-300/30"
                }`}
                title="Toggle Theme"
              >
                {isDark ? (
                  <Sun className="w-5 h-5" />
                ) : (
                  <Moon className="w-5 h-5" />
                )}
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-4 py-6">
        {workflow.step !== "greeting" && (
          <WorkflowProgress currentStep={workflow.step} isDark={isDark} />
        )}

        <div className="space-y-6 min-h-[calc(100vh-200px)]">
          {messages.map((message) => (
            <MessageComponent
              key={message.id}
              message={message}
              isDark={isDark}
            />
          ))}

          {isLoading && (
            <div className="flex gap-3">
              <div
                className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center transition-all duration-300 ${
                  isDark
                    ? "bg-purple-500/20 text-purple-300 shadow-lg shadow-purple-500/20"
                    : "bg-purple-100 text-purple-600 shadow-lg shadow-purple-200/50"
                }`}
              >
                <Bot className="w-4 h-4" />
              </div>
              <div
                className={`p-4 rounded-2xl transition-all duration-300 ${
                  isDark
                    ? "bg-gray-800/60 shadow-lg shadow-gray-900/20"
                    : "bg-white shadow-lg shadow-gray-200/50"
                }`}
              >
                <div className="flex items-center gap-2">
                  <Loader2
                    className={`w-4 h-4 animate-spin ${
                      isDark ? "text-purple-300" : "text-purple-600"
                    }`}
                  />
                  <span className={isDark ? "text-gray-300" : "text-gray-600"}>
                    Searching papers...
                  </span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* {/  Input Form  /}  */}
      <div
        className={`sticky bottom-0 backdrop-blur-xl border-t transition-all duration-300 ${
          isDark
            ? "bg-gray-900/80 border-gray-700"
            : "bg-white/80 border-gray-200"
        }`}
      >
        <div className="max-w-4xl mx-auto px-4 py-4">
          <form onSubmit={handleSubmit} className="relative">
            <div
              className={`flex items-center gap-3 p-3 rounded-3xl border transition-all duration-300 ${
                isDark
                  ? "bg-gray-800/90 border-gray-600 shadow-xl shadow-gray-900/20"
                  : "bg-white border-gray-300 shadow-xl shadow-gray-200/30"
              }`}
            >
              <input
                ref={inputRef}
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder={getInputPlaceholder()}
                className={`flex-1 bg-transparent outline-none text-sm transition-all duration-300 ${
                  isDark
                    ? "text-white placeholder-gray-400"
                    : "text-gray-800 placeholder-gray-500"
                }`}
              />
              <button
                type="submit"
                disabled={!inputValue.trim() || isLoading}
                className={`p-2 rounded-full transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed hover:scale-105 ${
                  !inputValue.trim() || isLoading
                    ? isDark
                      ? "bg-gray-700 text-gray-500"
                      : "bg-gray-200 text-gray-400"
                    : isDark
                    ? "bg-blue-500 text-white hover:bg-blue-600 shadow-lg shadow-blue-500/30"
                    : "bg-blue-500 text-white hover:bg-blue-600 shadow-lg shadow-blue-300/30"
                }`}
              >
                {isLoading ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default ResearchChatUI;
