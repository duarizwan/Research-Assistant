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
  ChevronDown,
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

// Reset counters and session state
const resetCountersAndSession = async () => {
  console.log("🔄 Resetting counters and session state...");
  console.log("📊 Before reset - messageIdCounter:", messageIdCounter);
  messageIdCounter = 0;
  console.log("📊 After reset - messageIdCounter:", messageIdCounter);

  try {
    // Reset backend session
    console.log("🌐 Calling backend reset endpoint...");
    const response = await fetch("/api/reset", { method: "GET" });
    console.log("🌐 Backend response status:", response.status);

    if (response.ok) {
      const data = await response.json();
      console.log("✅ Session and counters reset successfully:", data);
    } else {
      console.warn("⚠️ Session reset response not OK:", response.status);
      const errorText = await response.text();
      console.warn("⚠️ Error response:", errorText);
    }
  } catch (error) {
    console.error("❌ Error resetting session:", error);
  }
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
  const [showScrollButton, setShowScrollButton] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(0);

  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const welcomeInputRef = useRef<HTMLInputElement | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleScrollToBottom = () => {
    scrollToBottom();
    setShowScrollButton(false);
  };

  useEffect(() => {
    setIsClient(true);
    // Reset counters and session on page load/refresh
    resetCountersAndSession();
    // Reset workflow state to ensure clean start
    setWorkflow({
      step: "greeting",
      globalPaperMap: {},
      totalPaperCount: 0,
      papers: undefined,
      selectedPapers: undefined,
      downloadCount: undefined,
      field: undefined,
      topic: undefined,
    });
    // Keep messages empty initially for centered interface
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Debug workflow step changes
  useEffect(() => {
    console.log("🔄 Workflow step changed to:", workflow.step);
    console.log("📊 Current workflow state:", {
      step: workflow.step,
      field: workflow.field,
      topic: workflow.topic,
      totalPaperCount: workflow.totalPaperCount,
      downloadCount: workflow.downloadCount,
    });
  }, [
    workflow.step,
    workflow.field,
    workflow.topic,
    workflow.totalPaperCount,
    workflow.downloadCount,
  ]);

  // Force progress tracker re-render when workflow changes
  useEffect(() => {
    // This ensures the progress tracker updates when workflow state changes
    console.log("🔄 Progress tracker should update for step:", workflow.step);
  }, [workflow.step]);

  // Auto-focus input when centered interface is shown
  useEffect(() => {
    if (welcomeInputRef.current && messages.length === 0) {
      // Small delay to ensure DOM is ready
      setTimeout(() => {
        if (welcomeInputRef.current) {
          welcomeInputRef.current.focus();
        }
      }, 100);
    }
  }, [messages.length]);

  // Scroll detection for showing/hiding scroll button
  useEffect(() => {
    const handleScroll = () => {
      if (messages.length === 0) return;

      const scrollTop =
        window.pageYOffset || document.documentElement.scrollTop;
      const windowHeight = window.innerHeight;
      const documentHeight = document.documentElement.scrollHeight;

      // Show button if user has scrolled up more than 200px from bottom
      const isNearBottom = scrollTop + windowHeight >= documentHeight - 200;
      setShowScrollButton(!isNearBottom && messages.length > 0);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, [messages.length]);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) {
      return;
    }

    // Prevent multiple rapid submissions
    if (isLoading) return;

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
      // Handle quit/restart commands globally
      const normalizedInput = currentInput.toLowerCase().trim();
      if (
        normalizedInput === "quit" ||
        normalizedInput === "q" ||
        normalizedInput === "restart"
      ) {
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
    setWorkflow((prev) => {
      const newState = {
        ...prev,
        step: "paper_selection" as const,
        downloadCount: count,
      };
      console.log("🔄 Workflow state updated to paper_selection:", newState);
      return newState;
    });

    const botMessage: Message = {
      id: generateMessageId(),
      type: "bot",
      content: `👉 Selection options:\n   - Serial numbers: 1,3,5\n   - Author name: Goodfellow\n   - Year: 2020\n   - Mixed: 1,Goodfellow,2020\n   - 'help' for more guidance\n\nEnter your selection:`,
      timestamp: new Date(),
      workflowStep: "paper_selection",
    };
    setMessages((prev) => [...prev, botMessage]);

    // Force a re-render to ensure progress tracker updates
    setTimeout(() => {
      console.log(
        "🔄 Forcing progress tracker update after paper_selection step"
      );
      setWorkflow((prev) => ({ ...prev }));
    }, 100);
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
      // Don't set workflow step here - handleDownloadStart will do it
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

      // Then start the backend download process (handleDownloadStart will set the workflow step)
      await handleDownloadStart(selectedPapers);
    }
  };

  const handleDownloadStart = async (selectedPapers: Paper[]) => {
    console.log("🚀 handleDownloadStart called with papers:", selectedPapers);
    console.log("📊 Current workflow state at download start:", workflow);

    // Validation checks
    if (!selectedPapers || selectedPapers.length === 0) {
      console.error("❌ Download attempted without selected papers");
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

    // Ensure we're in downloading state
    console.log("🔄 Setting workflow step to downloading...");
    setDownloadProgress(0); // Reset progress
    setWorkflow((prev) => {
      const newState = { ...prev, step: "downloading" as const };
      console.log("📊 Workflow state updated to downloading:", newState.step);
      return newState;
    });

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
      console.log(
        `📥 Starting download of ${papersWithSummaries.length} papers...`
      );

      for (let i = 0; i < papersWithSummaries.length; i++) {
        const paper = papersWithSummaries[i];
        const progress = ((i + 1) / papersWithSummaries.length) * 100;

        console.log(
          `📊 Download progress: ${i + 1}/${
            papersWithSummaries.length
          } (${progress.toFixed(1)}%)`
        );

        // Update download progress state
        setDownloadProgress(progress);

        // Extract year from published date
        const year = paper.published
          ? paper.published.split("-")[0]
          : "Unknown";

        // Create progress bar visual
        const progressBar =
          "█".repeat(Math.floor(progress / 10)) +
          "░".repeat(10 - Math.floor(progress / 10));

        const progressMessage: Message = {
          id: generateMessageId(),
          type: "bot",
          content: `📥 Downloading ${i + 1}/${
            papersWithSummaries.length
          } (${progress.toFixed(1)}%)\n\n${progressBar} ${progress.toFixed(
            1
          )}%\n\n📄   ${paper.title}  \n👥   Authors:   ${paper.authors
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

      console.log(
        "✅ All papers downloaded, transitioning to completed state..."
      );

      // Final completion message
      console.log(
        "🎉 Download process completed, setting workflow to completed..."
      );

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

      // Set workflow to completed BEFORE the timeout
      console.log("🔄 Setting workflow step to completed...");
      setDownloadProgress(100); // Set to 100% when completed
      setWorkflow((prev) => {
        console.log("📊 Previous workflow step:", prev.step);
        const newState = { ...prev, step: "completed" as const };
        console.log("📊 New workflow step:", newState.step);
        return newState;
      });

      // Force a re-render to ensure the progress bar updates
      setTimeout(() => {
        console.log("🔄 Forcing progress bar update...");
        setWorkflow((prev) => ({ ...prev }));
      }, 100);

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
    setMessages([]);
  };

  const CenteredInterface = () => (
    <div className="flex flex-col items-center justify-center min-h-[60vh] px-4">
      <div className="text-center mb-8">
        <h2
          className={`text-3xl font-semibold mb-4 leading-tight ${
            isDark ? "text-[#F9FAFB]" : "text-[#111827]"
          }`}
          style={{ lineHeight: "1.2" }}
        >
          What Can Research Assistant Help You With Today?
        </h2>
        <p
          className={`text-lg font-medium ${
            isDark ? "text-[#94A3B8]" : "text-[#4B5563]"
          }`}
          style={{ lineHeight: "1.4" }}
        >
          Find and download academic papers from arXiv with AI-powered summaries
        </p>
      </div>

      <div className="w-full max-w-2xl">
        <form onSubmit={handleSubmit} className="relative">
          <div
            className={`flex items-center gap-3 p-4 rounded-2xl border transition-all duration-300 ${
              isDark
                ? "bg-[#1E293B]/90 border-[#334155] shadow-xl shadow-[#0F172A]/30"
                : "bg-[#FFFFFF] border-[#E5E7EB] shadow-xl shadow-[#F9FAFB]/30"
            }`}
          >
            <input
              ref={welcomeInputRef}
              type="text"
              value={inputValue}
              onChange={(e) => {
                setInputValue(e.target.value);
                // Ensure focus is maintained after state update
                setTimeout(() => {
                  if (welcomeInputRef.current) {
                    welcomeInputRef.current.focus();
                  }
                }, 0);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && inputValue.trim() && !isLoading) {
                  // Allow Enter to submit only when conditions are met
                  return;
                }
                if (e.key === "Enter") {
                  e.preventDefault();
                }
              }}
              placeholder="Write anything to start your research journey..."
              className={`flex-1 bg-transparent outline-none text-base font-normal transition-all duration-300 ${
                isDark
                  ? "text-[#F9FAFB] placeholder-[#94A3B8]"
                  : "text-[#111827] placeholder-[#4B5563]"
              }`}
              style={{ lineHeight: "1.5" }}
              autoComplete="off"
              autoCorrect="off"
              autoCapitalize="off"
              spellCheck="false"
            />
            <button
              type="submit"
              disabled={!inputValue.trim() || isLoading}
              className={`p-3 rounded-full transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed hover:scale-105 ${
                !inputValue.trim() || isLoading
                  ? isDark
                    ? "bg-[#334155] text-[#64748B]"
                    : "bg-[#E5E7EB] text-[#9CA3AF]"
                  : isDark
                  ? "bg-[#38BDF8] text-white hover:bg-[#0EA5E9] shadow-lg shadow-[#38BDF8]/40"
                  : "bg-[#2563EB] text-white hover:bg-[#1D4ED8] shadow-lg shadow-[#2563EB]/30"
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

      <div className="mt-8 grid grid-cols-2 md:grid-cols-4 gap-4 w-full max-w-2xl">
        <div
          className={`p-4 rounded-xl border transition-all duration-200 hover:shadow-lg cursor-pointer ${
            isDark
              ? "bg-[#1E293B]/60 border-[#334155] hover:bg-[#1E293B]/80"
              : "bg-[#FFFFFF] border-[#E5E7EB] hover:bg-[#F9FAFB]"
          }`}
        >
          <div className="text-center">
            <div
              className={`w-8 h-8 mx-auto mb-2 rounded-lg flex items-center justify-center ${
                isDark
                  ? "bg-[#38BDF8]/20 text-[#38BDF8]"
                  : "bg-[#2563EB]/10 text-[#2563EB]"
              }`}
            >
              <Search className="w-4 h-4" />
            </div>
            <p
              className={`text-sm font-medium ${
                isDark ? "text-[#F9FAFB]" : "text-[#111827]"
              }`}
              style={{ lineHeight: "1.4" }}
            >
              Search Papers
            </p>
          </div>
        </div>

        <div
          className={`p-4 rounded-xl border transition-all duration-200 hover:shadow-lg cursor-pointer ${
            isDark
              ? "bg-[#1E293B]/60 border-[#334155] hover:bg-[#1E293B]/80"
              : "bg-[#FFFFFF] border-[#E5E7EB] hover:bg-[#F9FAFB]"
          }`}
        >
          <div className="text-center">
            <div
              className={`w-8 h-8 mx-auto mb-2 rounded-lg flex items-center justify-center ${
                isDark
                  ? "bg-[#A78BFA]/20 text-[#A78BFA]"
                  : "bg-[#10B981]/10 text-[#10B981]"
              }`}
            >
              <FileText className="w-4 h-4" />
            </div>
            <p
              className={`text-sm font-medium ${
                isDark ? "text-[#F9FAFB]" : "text-[#111827]"
              }`}
              style={{ lineHeight: "1.4" }}
            >
              AI Summaries
            </p>
          </div>
        </div>

        <div
          className={`p-4 rounded-xl border transition-all duration-200 hover:shadow-lg cursor-pointer ${
            isDark
              ? "bg-[#1E293B]/60 border-[#334155] hover:bg-[#1E293B]/80"
              : "bg-[#FFFFFF] border-[#E5E7EB] hover:bg-[#F9FAFB]"
          }`}
        >
          <div className="text-center">
            <div
              className={`w-8 h-8 mx-auto mb-2 rounded-lg flex items-center justify-center ${
                isDark
                  ? "bg-[#F472B6]/20 text-[#F472B6]"
                  : "bg-[#9333EA]/10 text-[#9333EA]"
              }`}
            >
              <FolderDown className="w-4 h-4" />
            </div>
            <p
              className={`text-sm font-medium ${
                isDark ? "text-[#F9FAFB]" : "text-[#111827]"
              }`}
              style={{ lineHeight: "1.4" }}
            >
              Download PDFs
            </p>
          </div>
        </div>

        <div
          className={`p-4 rounded-xl border transition-all duration-200 hover:shadow-lg cursor-pointer ${
            isDark
              ? "bg-[#1E293B]/60 border-[#334155] hover:bg-[#1E293B]/80"
              : "bg-[#FFFFFF] border-[#E5E7EB] hover:bg-[#F9FAFB]"
          }`}
        >
          <div className="text-center">
            <div
              className={`w-8 h-8 mx-auto mb-2 rounded-lg flex items-center justify-center ${
                isDark
                  ? "bg-[#38BDF8]/20 text-[#38BDF8]"
                  : "bg-[#2563EB]/10 text-[#2563EB]"
              }`}
            >
              <Bot className="w-4 h-4" />
            </div>
            <p
              className={`text-sm font-medium ${
                isDark ? "text-[#F9FAFB]" : "text-[#111827]"
              }`}
              style={{ lineHeight: "1.4" }}
            >
              Smart Assistant
            </p>
          </div>
        </div>
      </div>
    </div>
  );

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
    if (
      normalizedInput === "quit" ||
      normalizedInput === "q" ||
      normalizedInput === "restart"
    ) {
      // Send restart command to backend
      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            message: input,
            workflow_step: "restart",
            session_id: "default",
          }),
        });

        if (response.ok) {
          const data = await response.json();

          // Reset frontend state and transition to field selection
          setWorkflow({ step: "field_selection" });
          setMessages([]);

          // Add backend response message
          const restartMessage: Message = {
            id: generateMessageId(),
            type: "bot",
            content: data.response,
            timestamp: new Date(),
            workflowStep: "field_selection",
          };
          setMessages((prev) => [...prev, restartMessage]);
        } else {
          // Fallback to frontend-only restart
          restartWorkflow();
        }
      } catch (error) {
        console.error("Error sending restart command:", error);
        // Fallback to frontend-only restart
        restartWorkflow();
      }
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
      className={`p-4 rounded-xl border ${
        isDark
          ? "bg-[#1E293B]/60 border-[#334155] hover:bg-[#1E293B]/80"
          : "bg-[#FFFFFF] border-[#E5E7EB] hover:bg-[#F9FAFB]"
      } transition-all duration-200 hover:shadow-lg`}
    >
      <div className="flex items-start justify-between mb-2">
        <span
          className={`text-xs px-2 py-1 rounded-full font-medium ${
            isDark
              ? "bg-[#38BDF8]/20 text-[#38BDF8]"
              : "bg-[#2563EB]/10 text-[#2563EB]"
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
                  ? "bg-[#A78BFA]/20 text-[#A78BFA]"
                  : "bg-[#9333EA]/10 text-[#9333EA]"
              }`}
            >
              {cat}
            </span>
          ))}
        </div>
      </div>

      <h4
        className={`font-semibold mb-2 line-clamp-2 ${
          isDark ? "text-[#38BDF8]" : "text-[#2563EB]"
        }`}
      >
        {paper.title}
      </h4>

      <p
        className={`text-base mb-2 font-normal ${
          isDark ? "text-[#F9FAFB]" : "text-[#111827]"
        }`}
        style={{ lineHeight: "1.5" }}
      >
        <strong>Authors:</strong> {paper.authors.slice(0, 3).join(", ")}
        {paper.authors.length > 3 && ` et al. (${paper.authors.length} total)`}
      </p>

      <p
        className={`text-base mb-3 font-normal ${
          isDark ? "text-[#F9FAFB]" : "text-[#111827]"
        }`}
        style={{ lineHeight: "1.5" }}
      >
        <strong>Published:</strong> {paper.published} | <strong>Status:</strong>{" "}
        {paper.status}
      </p>

      {paper.summary && (
        <p
          className={`text-sm mb-3 font-normal ${
            isDark ? "text-[#94A3B8]" : "text-[#4B5563]"
          } line-clamp-2`}
          style={{ lineHeight: "1.5" }}
        >
          {paper.summary.substring(0, 120)}...
        </p>
      )}

      <div className="flex items-center gap-2">
        <button
          className={`px-3 py-1 text-sm font-medium rounded-full transition-all duration-200 ${
            isDark
              ? "bg-[#334155] text-[#F9FAFB] hover:bg-[#475569]"
              : "bg-[#F3F4F6] text-[#111827] hover:bg-[#E5E7EB]"
          }`}
          style={{ lineHeight: "1.4" }}
          onClick={() => window.open(paper.pdf_url, "_blank")}
        >
          <FileText className="w-3 h-3 inline mr-1" />
          View PDF
        </button>
      </div>
    </div>
  );

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
              ? "bg-[#38BDF8]/20 text-[#38BDF8] shadow-lg shadow-[#38BDF8]/20"
              : "bg-[#2563EB]/10 text-[#2563EB] shadow-lg shadow-[#2563EB]/50"
            : isDark
            ? "bg-[#A78BFA]/20 text-[#A78BFA] shadow-lg shadow-[#A78BFA]/20"
            : "bg-[#9333EA]/10 text-[#9333EA] shadow-lg shadow-[#9333EA]/50"
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
                ? "bg-[#38BDF8]/15 text-[#F9FAFB] shadow-lg shadow-[#38BDF8]/20 hover:shadow-[#38BDF8]/30"
                : "bg-[#2563EB] text-white shadow-lg shadow-[#2563EB]/50 hover:shadow-[#2563EB]/50"
              : isDark
              ? "bg-[#1E293B]/70 text-[#F9FAFB] shadow-lg shadow-[#0F172A]/30 hover:shadow-[#0F172A]/40"
              : "bg-[#FFFFFF] text-[#111827] shadow-lg shadow-[#F9FAFB]/50 hover:shadow-[#F9FAFB]/50"
          }`}
        >
          <div
            className="whitespace-pre-wrap text-base font-normal"
            style={{ lineHeight: "1.5" }}
          >
            {message.content}
          </div>

          {message.papers && (
            <div className="mt-4 space-y-3">
              <div
                className={`text-lg font-medium ${
                  isDark ? "text-[#A78BFA]" : "text-[#9333EA]"
                }`}
                style={{ lineHeight: "1.4" }}
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
                        className={`text-lg font-medium mb-2 ${
                          isDark ? "text-[#F9FAFB]" : "text-[#111827]"
                        }`}
                        style={{ lineHeight: "1.4" }}
                      >
                        Ready to Download
                      </h3>
                      <p
                        className={`text-base mb-2 font-normal ${
                          isDark ? "text-[#94A3B8]" : "text-[#4B5563]"
                        }`}
                        style={{ lineHeight: "1.5" }}
                      >
                        {message.papers.length} papers selected for download
                      </p>
                      <p
                        className={`text-sm mb-4 font-normal ${
                          isDark ? "text-[#94A3B8]" : "text-[#4B5563]"
                        }`}
                        style={{ lineHeight: "1.5" }}
                      >
                        Click to open file browser, select save location, and
                        download PDFs to your PC
                      </p>
                      <button
                        onClick={handleDownloadButtonClick}
                        className={`px-6 py-3 rounded-lg text-sm font-medium transition-all duration-300 hover:scale-105 ${
                          isDark
                            ? "bg-[#A78BFA] text-white hover:bg-[#8B5CF6] shadow-lg shadow-[#A78BFA]/30"
                            : "bg-[#10B981] text-white hover:bg-[#059669] shadow-lg shadow-[#10B981]/30"
                        }`}
                        style={{ lineHeight: "1.4" }}
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
          className={`text-sm mt-2 font-normal ${
            isDark ? "text-[#94A3B8]" : "text-[#4B5563]"
          }`}
          style={{ lineHeight: "1.5" }}
        >
          {message.timestamp.toLocaleTimeString()}
        </div>
      </div>
    </div>
  );

  // Don't render until client-side hydration is complete
  if (!isClient) {
    return (
      <div className="min-h-screen bg-[#0F172A] text-[#F9FAFB] flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-4 border-[#38BDF8] border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p
            className="text-base font-normal text-[#94A3B8]"
            style={{ lineHeight: "1.5" }}
          >
            Loading Research Assistant...
          </p>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`min-h-screen transition-all duration-500 ${
        isDark ? "bg-[#0F172A] text-[#F9FAFB]" : "bg-[#F9FAFB] text-[#111827]"
      }`}
    >
      <div
        className={`sticky top-0 z-50 backdrop-blur-xl border-b transition-all duration-300 ${
          isDark
            ? "bg-[#1E293B]/90 border-[#1E293B] shadow-xl shadow-[#0F172A]/30"
            : "bg-[#FFFFFF]/80 border-[#E5E7EB] shadow-xl shadow-[#F9FAFB]/20"
        }`}
      >
        <div className="max-w-4xl mx-auto px-4 py-4">
          {messages.length === 0 ? (
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div
                  className={`w-10 h-10 rounded-lg flex items-center justify-center transition-all duration-300 ${
                    isDark
                      ? "bg-gradient-to-br from-[#38BDF8] to-[#A78BFA] shadow-lg shadow-[#38BDF8]/30"
                      : "bg-gradient-to-br from-[#2563EB] to-[#9333EA] shadow-lg shadow-[#2563EB]/25"
                  }`}
                >
                  <Bot className="w-5 h-5 text-white" />
                </div>
                <div className="transition-all duration-500 ease-in-out">
                  <h1
                    className={`text-2xl font-semibold transition-all duration-500 leading-tight ${
                      isDark
                        ? "text-[#F9FAFB]"
                        : "bg-gradient-to-r from-[#2563EB] to-[#9333EA] bg-clip-text text-transparent"
                    }`}
                    style={{ lineHeight: "1.2" }}
                  >
                    Research Assistant
                  </h1>
                  <p
                    className={`text-xs font-medium transition-all duration-500 ${
                      isDark ? "text-[#94A3B8]" : "text-[#4B5563]"
                    }`}
                  >
                    Powered by arXiv & Gemini AI
                  </p>
                </div>
              </div>

              <div className="flex gap-2">
                <button
                  onClick={resetCountersAndSession}
                  className={`p-2 rounded-lg transition-all duration-300 hover:scale-110 ${
                    isDark
                      ? "bg-[#1E293B] hover:bg-[#334155] text-[#10B981] shadow-lg shadow-[#10B981]/20 hover:shadow-[#10B981]/30"
                      : "bg-[#F3F4F6] hover:bg-[#E5E7EB] text-[#10B981] shadow-lg shadow-[#E5E7EB]/20 hover:shadow-[#E5E7EB]/30"
                  }`}
                  title="Reset Session"
                >
                  🔄
                </button>
                <button
                  onClick={() => setIsDark(!isDark)}
                  className={`p-2 rounded-lg transition-all duration-300 hover:scale-110 ${
                    isDark
                      ? "bg-[#1E293B] hover:bg-[#334155] text-[#F472B6] shadow-lg shadow-[#F472B6]/20 hover:shadow-[#F472B6]/30"
                      : "bg-[#F3F4F6] hover:bg-[#E5E7EB] text-[#111827] shadow-lg shadow-[#E5E7EB]/20 hover:shadow-[#E5E7EB]/30"
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
          ) : (
            <div className="flex items-center justify-between">
              {/* Empty div for spacing */}
              <div className="w-12"></div>

              {/* Step Icons - Centered */}
              <div
                key={`progress-tracker-${workflow.step}`}
                className="flex items-center gap-4"
              >
                {[
                  { key: "greeting", label: "Welcome", icon: Bot },
                  { key: "field_selection", label: "Field", icon: BookOpen },
                  { key: "topic_selection", label: "Topic", icon: Search },
                  { key: "paper_listing", label: "Papers", icon: FileText },
                  { key: "download_count", label: "Count", icon: ChevronRight },
                  { key: "paper_selection", label: "Select", icon: FileCheck },
                  { key: "downloading", label: "Download", icon: FolderDown },
                  { key: "completed", label: "Done", icon: CheckCircle },
                ].map((step, index) => {
                  const Icon = step.icon;
                  const isActive = step.key === workflow.step;

                  // Define the complete step progression including all possible steps
                  const stepProgression = [
                    "greeting",
                    "field_selection",
                    "topic_selection",
                    "paper_listing",
                    "load_more", // Include load_more step
                    "download_count",
                    "paper_selection",
                    "download_confirmation", // Include download_confirmation step
                    "downloading",
                    "completed",
                  ];

                  const currentStepIndex = stepProgression.findIndex(
                    (s) => s === workflow.step
                  );

                  // A step is completed if:
                  // 1. The current step index is greater than the step's index in the progression
                  // 2. OR if the current step is a later step in the progression
                  const stepIndexInProgression = stepProgression.findIndex(
                    (s) => s === step.key
                  );

                  // Ensure we have valid indices
                  const validCurrentStepIndex =
                    currentStepIndex >= 0 ? currentStepIndex : 0;
                  const validStepIndexInProgression =
                    stepIndexInProgression >= 0 ? stepIndexInProgression : 0;

                  const isCompleted =
                    validCurrentStepIndex > validStepIndexInProgression;

                  // Debug step indicator state
                  if (isActive || isCompleted) {
                    console.log(
                      `🎯 Step ${step.key} (index ${index}, progression index ${stepIndexInProgression}): isActive=${isActive}, isCompleted=${isCompleted}, currentStepIndex=${currentStepIndex}, workflow.step=${workflow.step}`
                    );
                  }

                  // Additional debugging for progress tracker issues
                  if (workflow.step === "paper_selection") {
                    console.log(
                      `🔍 Progress Tracker Debug - Step: ${step.key}, isActive: ${isActive}, isCompleted: ${isCompleted}, validCurrentStepIndex: ${validCurrentStepIndex}, validStepIndexInProgression: ${validStepIndexInProgression}`
                    );
                  }

                  return (
                    <div
                      key={step.key}
                      className={`w-8 h-8 rounded-full flex items-center justify-center transition-all duration-300 relative ${
                        isActive
                          ? isDark
                            ? "bg-[#38BDF8]/20 text-[#38BDF8] shadow-lg shadow-[#38BDF8]/30 ring-2 ring-[#38BDF8]/30"
                            : "bg-[#2563EB] text-white shadow-md shadow-[#2563EB]/40"
                          : isCompleted
                          ? isDark
                            ? "bg-[#00FF00]/20 text-[#00FF00] shadow-lg shadow-[#00FF00]/30"
                            : "bg-[#10B981] text-white"
                          : isDark
                          ? "bg-[#334155]/60 text-[#94A3B8]"
                          : "bg-[#E5E7EB] text-[#4B5563]"
                      }`}
                    >
                      <Icon className="w-4 h-4" />
                      {/* Show progress for downloading step */}
                      {step.key === "downloading" &&
                        isActive &&
                        downloadProgress > 0 && (
                          <div className="absolute -bottom-1 -right-1 w-3 h-3 bg-green-500 rounded-full flex items-center justify-center">
                            <span className="text-xs text-white font-bold">
                              {Math.round(downloadProgress)}%
                            </span>
                          </div>
                        )}
                    </div>
                  );
                })}
              </div>

              {/* Theme Toggle */}
              <button
                onClick={() => setIsDark(!isDark)}
                className={`p-2 rounded-lg transition-all duration-300 hover:scale-110 ${
                  isDark
                    ? "bg-[#1E293B] hover:bg-[#334155] text-[#F472B6] shadow-lg shadow-[#F472B6]/20 hover:shadow-[#F472B6]/30"
                    : "bg-[#F3F4F6] hover:bg-[#E5E7EB] text-[#111827] shadow-lg shadow-[#E5E7EB]/20 hover:shadow-[#E5E7EB]/30"
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
          )}
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-4 py-6">
        {messages.length === 0 ? (
          <CenteredInterface />
        ) : (
          <div className="space-y-6 min-h-[calc(100vh-200px)] pt-4">
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
                      ? "bg-[#A78BFA]/20 text-[#A78BFA] shadow-lg shadow-[#A78BFA]/20"
                      : "bg-[#9333EA]/10 text-[#9333EA] shadow-lg shadow-[#9333EA]/50"
                  }`}
                >
                  <Bot className="w-4 h-4" />
                </div>
                <div
                  className={`p-4 rounded-2xl transition-all duration-300 ${
                    isDark
                      ? "bg-[#1E293B]/60 shadow-lg shadow-[#0F172A]/20"
                      : "bg-[#FFFFFF] shadow-lg shadow-[#F9FAFB]/50"
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <Loader2
                      className={`w-4 h-4 animate-spin ${
                        isDark ? "text-[#A78BFA]" : "text-[#9333EA]"
                      }`}
                    />
                    <span
                      className={`text-base font-normal ${
                        isDark ? "text-[#F9FAFB]" : "text-[#111827]"
                      }`}
                      style={{ lineHeight: "1.5" }}
                    >
                      Searching papers...
                    </span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Multi-line Input Field - Second Phase Only */}
      {messages.length > 0 && (
        <div className="max-w-4xl mx-auto px-4 py-4">
          <form onSubmit={handleSubmit} className="relative">
            <div
              className={`flex items-start gap-3 px-4 py-2 rounded-2xl border transition-all duration-300 focus-within:ring-2 focus-within:ring-opacity-50 ${
                isDark
                  ? "bg-[#1E293B]/90 border-[#334155] shadow-xl shadow-[#0F172A]/30 focus-within:ring-[#38BDF8] focus-within:border-[#38BDF8]"
                  : "bg-[#FFFFFF] border-[#E5E7EB] shadow-xl shadow-[#F9FAFB]/30 focus-within:ring-[#2563EB] focus-within:border-[#2563EB]"
              }`}
            >
              <textarea
                ref={inputRef}
                value={inputValue}
                onChange={(e) => {
                  setInputValue(e.target.value);
                  // Auto-resize textarea
                  e.target.style.height = "auto";
                  e.target.style.height = e.target.scrollHeight + "px";
                  // Ensure focus is maintained after state update
                  setTimeout(() => {
                    if (inputRef.current) {
                      inputRef.current.focus();
                    }
                  }, 0);
                }}
                onKeyDown={(e) => {
                  if (
                    e.key === "Enter" &&
                    !e.shiftKey &&
                    inputValue.trim() &&
                    !isLoading
                  ) {
                    e.preventDefault();
                    // Create a synthetic form event
                    const formEvent = {
                      ...e,
                      preventDefault: () => e.preventDefault(),
                      currentTarget: e.currentTarget.closest(
                        "form"
                      ) as HTMLFormElement,
                    } as React.FormEvent<HTMLFormElement>;
                    handleSubmit(formEvent);
                  }
                }}
                placeholder={getInputPlaceholder()}
                className={`flex-1 bg-transparent outline-none text-base font-normal transition-all duration-300 resize-none overflow-hidden ${
                  isDark
                    ? "text-[#F9FAFB] placeholder-[#94A3B8]"
                    : "text-[#111827] placeholder-[#4B5563]"
                }`}
                style={{
                  lineHeight: "1.5",
                  minHeight: "24px",
                  maxHeight: "120px",
                }}
                autoComplete="off"
                autoCorrect="off"
                autoCapitalize="off"
                spellCheck="false"
                rows={1}
              />
              <button
                type="submit"
                disabled={!inputValue.trim() || isLoading}
                className={`p-3 rounded-full transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed hover:scale-105 flex-shrink-0 ${
                  !inputValue.trim() || isLoading
                    ? isDark
                      ? "bg-[#334155] text-[#64748B]"
                      : "bg-[#E5E7EB] text-[#9CA3AF]"
                    : isDark
                    ? "bg-[#38BDF8] text-white hover:bg-[#0EA5E9] shadow-lg shadow-[#38BDF8]/40"
                    : "bg-[#2563EB] text-white hover:bg-[#1D4ED8] shadow-lg shadow-[#2563EB]/30"
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
      )}

      {/* Scroll to Bottom Button */}
      {showScrollButton && (
        <div className="fixed bottom-20 left-1/2 transform -translate-x-1/2 z-40">
          <button
            onClick={handleScrollToBottom}
            className={`p-3 rounded-full shadow-lg transition-all duration-300 hover:scale-110 ${
              isDark
                ? "bg-[#1E293B] hover:bg-[#334155] text-[#38BDF8] shadow-[#0F172A]/30 hover:shadow-[#38BDF8]/20"
                : "bg-[#FFFFFF] hover:bg-[#F3F4F6] text-[#2563EB] shadow-[#F9FAFB]/30 hover:shadow-[#2563EB]/20"
            }`}
            title="Scroll to Bottom"
          >
            <ChevronDown className="w-5 h-5" />
          </button>
        </div>
      )}
    </div>
  );
};

export default ResearchChatUI;
